using System;
using System.IO;
using System.Linq;
using GoodAI.Core;
using GoodAI.Core.Memory;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Runtime.InteropServices;

namespace GoodAI.Modules.Transforms
{
    public enum ReductionMode
    {
        i_Sum_i, i_MinIdx_2i, i_MaxIdx_2i, i_MinIdxMaxIdx_4i, c_Sum_c,
        f_Sum_f, f_MinIdx_fi, f_MinIdx_ff, f_MaxIdx_fi, f_MaxIdx_ff, f_MinIdxMaxIdx_fifi, f_MinMax_2f
    };

    public enum ProductMode
    {
        f_DotProduct_f, i_DotProduct_i, f_Cosine_f, c_ComplexDot_c
    };

    public class MyReductionKernel<T> where T : struct
    {
        private MyCudaKernel m_kernel;
        CUdeviceptr CUdeviceNullptr = new CUdeviceptr(0);

        private int TSize;
        private int m_outTypeSize;
        private int m_inTypeSize;

        public bool segmented;
        public bool distributed;
        public int timeOffset;
        public int outOffset;
        public int inOffset;
        public int stride;
        int nGPU;

        public MyReductionKernel(int nGPU, ReductionMode mode, bool segmented = false, bool distributed = false)
        {
            switch (mode)
            {
                case ReductionMode.i_Sum_i: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7i_Sum_iiLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 4; break;
                case ReductionMode.i_MinIdx_2i: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MinIdx_2iiLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.i_MaxIdx_2i: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MaxIdx_2iiLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.i_MinIdxMaxIdx_4i: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 16; break;
                case ReductionMode.f_Sum_f: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7f_Sum_ffLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 4; break;
                case ReductionMode.f_MinMax_2f: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinMax_2ffLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.f_MinIdx_fi: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fifLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.f_MaxIdx_fi: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fifLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.f_MinIdxMaxIdx_fifi: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 16; break;
                case ReductionMode.f_MinIdx_ff: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fffLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.f_MaxIdx_ff: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fffLj512EEvPvPVKvS1_jjjjb"); m_inTypeSize = 4; m_outTypeSize = 8; break;
                case ReductionMode.c_Sum_c: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7c_Sum_c7ComplexLj512EEvPvPVKvS2_jjjjb"); m_inTypeSize = 8; m_outTypeSize = 8; break;
                default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
            }

            TSize = Marshal.SizeOf(typeof(T));
            if (TSize > m_inTypeSize || TSize > m_outTypeSize || TSize % m_inTypeSize != 0 ||
                TSize % m_outTypeSize != 0 || m_inTypeSize == 0 || m_outTypeSize == 0)
                throw new ArgumentOutOfRangeException("mode", "MemoryBlock type can be incompatible with reduction in/out types.");

            this.stride = 1;
            this.segmented = segmented;
            this.distributed = distributed;
            this.nGPU = nGPU;
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input)
        {
            // rDeviceVar != null ? rDeviceVar.DevicePointer + offset * rDeviceVar.TypeSize : default(CUdeviceptr);
            CUdeviceptr outputPtr = output.GetDevicePtr(nGPU, outOffset * (m_outTypeSize / TSize));//, timeOffset);
            CUdeviceptr inputPtr = input.GetDevicePtr(nGPU, inOffset * (m_inTypeSize / TSize));//, timeOffset);

            dim3 blockDims = new dim3(512);
            dim3 gridDims = new dim3(1);
            m_kernel.SetupExecution(blockDims, gridDims);
            m_kernel.DynamicSharedMemory = 512 * (uint)m_outTypeSize;

            // void* rawOut, volatile const void* rawIn, void* tempBuffer, unsigned int size, unsigned int outOff, unsigned int inOff, unsigned int stride, bool segmented
            m_kernel.Run(outputPtr, inputPtr, 0, input.Count, 0, 0, stride, Convert.ToInt32(segmented));
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input, MyMemoryBlock<float> buffer)
        {
            // rDeviceVar != null ? rDeviceVar.DevicePointer + offset * rDeviceVar.TypeSize : default(CUdeviceptr);
            CUdeviceptr outputPtr = output.GetDevicePtr(nGPU, outOffset * (m_outTypeSize / TSize), timeOffset);
            CUdeviceptr inputPtr = output.GetDevicePtr(nGPU, inOffset * (m_inTypeSize / TSize), timeOffset);

            // void* rawOut, volatile const void* rawIn, void* tempBuffer, unsigned int size, unsigned int outOff, unsigned int inOff, unsigned int stride, bool segmented
            m_kernel.RunAsync(stream, outputPtr, inputPtr, buffer, input.Count, 0, 0, 1, Convert.ToInt32(segmented));
        }

        protected void SetupExecution(int numOfParallelUnits) { }
        protected virtual void SetupExecution(dim3 blockDimensions, dim3 gridDimensions) { }
    }

    public class MyProductKernel<T> where T : struct
    {
        private MyCudaKernel m_kernel;

        private int TSize;
        private int m_outTypeSize;
        private int m_inTypeSize;

        public bool segmented;
        public bool distributed;
        public int timeOffset;
        public int outOffset;
        public int inOffset;
        public int stride;

        int nGPU;

        public MyProductKernel(int nGPU, ProductMode mode, bool segmented = false, bool distributed = false)
        {
            switch (mode)
            {
                case ProductMode.i_DotProduct_i: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7i_Dot_iiLj512EEvPvjPVKvS3_S1_jbb"); m_inTypeSize = 4; m_outTypeSize = 4; break;
                case ProductMode.f_DotProduct_f: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7f_Dot_ffLj512EEvPvjPVKvS3_S1_jbb"); m_inTypeSize = 4; m_outTypeSize = 4; break;
                case ProductMode.f_Cosine_f: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI10f_Cosine_ffLj512EEvPvjPVKvS3_S1_jbb"); m_inTypeSize = 4; m_outTypeSize = 16; break;
                case ProductMode.c_ComplexDot_c: m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI14c_ComplexDot_c7ComplexLj512EEvPvjPVKvS4_S2_jbb"); m_inTypeSize = 8; m_outTypeSize = 8; break;
                default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
            }

            TSize = Marshal.SizeOf(typeof(T));
            if (TSize > m_inTypeSize || TSize > m_outTypeSize || TSize % m_inTypeSize != 0 ||
                TSize % m_outTypeSize != 0 || m_inTypeSize == 0 || m_outTypeSize == 0)
                throw new ArgumentOutOfRangeException("mode", "MemoryBlock type can be incompatible with reduction in/out types.");

            this.segmented = segmented;
            this.distributed = distributed;
            this.nGPU = nGPU;
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            CUdeviceptr outputPtr = output.GetDevicePtr(nGPU, outOffset * (m_outTypeSize / TSize), timeOffset);

            //void* rawOut, unsigned int outOff, volatile const void* rawIn1, volatile const void* rawIn2, void* tempBuffer, unsigned int size, bool segmented, bool distributed
            m_kernel.Run(outputPtr, 0, input1, input2, 0, input2.Count, Convert.ToInt32(segmented), Convert.ToInt32(distributed));
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2, MyMemoryBlock<T> buffer)
        {
            CUdeviceptr outputPtr = output.GetDevicePtr(nGPU, outOffset * (m_outTypeSize / TSize), timeOffset);

            //MyMemoryManager.Instance.CreateMemoryBlock<float>()

            //void* rawOut, unsigned int outOff, volatile const void* rawIn1, volatile const void* rawIn2, void* tempBuffer, unsigned int size, bool segmented, bool distributed
            m_kernel.RunAsync(stream, outputPtr, buffer, input1, input2, 0, input2.Count, Convert.ToInt32(segmented), Convert.ToInt32(distributed));
        }

        protected void SetupExecution(int numOfParallelUnits) { }
        protected virtual void SetupExecution(dim3 blockDimensions, dim3 gridDimensions) { }
    }

    /// <summary>
    /// OBSOLETE WILL BE DELETED SOON
    /// </summary>
    public static class MyReductionFactory
    {
        public enum Mode
        {
            i_Sum_i, i_MinIdx_2i, i_MaxIdx_2i, i_MinIdxMaxIdx_4i, c_Sum_c,
            f_Sum_f, f_MinIdx_fi, f_MinIdx_ff, f_MaxIdx_fi, f_MaxIdx_ff, f_MinIdxMaxIdx_fifi, f_MinMax_2f,
            f_DotProduct_f, i_DotProduct_i, f_Cosine_f, c_ComplexDot_c
        };

        /// <summary>
        /// OBSOLETE WILL BE DELETED SOON
        /// </summary>
        public static MyCudaKernel Kernel(int nGPU, Mode mode, int blockCount = 1, bool segmented = false, bool distributed = false)
        {
            uint outSize = 0;
            MyCudaKernel kernel = null;

            if (blockCount < 1)
                throw new ArgumentOutOfRangeException("blockCount", "Invalid block count -- must be positive.");

            if (!segmented)
            {
                switch (mode)
                {
                    case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7i_Sum_iiLj512EEvPvPVKvjjjjb"); outSize = 4; break;
                    case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MinIdx_2iiLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MaxIdx_2iiLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvjjjjb"); outSize = 16; break;
                    case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7f_Sum_ffLj512EEvPvPVKvjjjjb"); outSize = 4; break;
                    case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fifLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fffLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fifLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fffLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinMax_2ffLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvjjjjb"); outSize = 16; break;
                    case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7i_Dot_iiLj512EEvPvjPVKvS3_jb"); outSize = 4; break;
                    case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7f_Dot_ffLj512EEvPvjPVKvS3_jb"); outSize = 4; break;
                    case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI10f_Cosine_ffLj512EEvPvjPVKvS3_jb"); outSize = 16; break;
                    case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI14c_ComplexDot_c7ComplexLj512EEvPvjPVKvS4_jb"); outSize = 8; break;
                    case Mode.c_Sum_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7c_Sum_c7ComplexLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                }
            }
            else // segmented
            {
                if (!distributed)
                {
                    switch (mode)
                    {
                        case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7i_Sum_iiLj512EEvPvPVKvjj"); outSize = 4; break;
                        case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MinIdx_2iiLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MaxIdx_2iiLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvjj"); outSize = 16; break;
                        case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7f_Sum_ffLj512EEvPvPVKvjj"); outSize = 4; break;
                        case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fifLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fffLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fifLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fffLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinMax_2ffLj512EEvPvPVKvjj"); outSize = 8; break;
                        case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvjj"); outSize = 16; break;
                        case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7i_Dot_iiLj512EEvPvPVKvS3_j"); outSize = 4; break;
                        case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7f_Dot_ffLj512EEvPvPVKvS3_j"); outSize = 4; break;
                        case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI10f_Cosine_ffLj512EEvPvPVKvS3_j"); outSize = 16; break;
                        case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI14c_ComplexDot_c7ComplexLj512EEvPvPVKvS4_j"); outSize = 8; break;
                        case Mode.c_Sum_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7c_Sum_c7ComplexLj512EEvPvPVKvjj"); outSize = 8; break;
                        default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                    }
                }
                else // distributed
                {
                    switch (mode)
                    {
                        case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI7i_Dot_iiLj512EEvPvPVKvS3_j"); outSize = 4; break;
                        case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI7f_Dot_ffLj512EEvPvPVKvS3_j"); outSize = 4; break;
                        case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI10f_Cosine_ffLj512EEvPvPVKvS3_j"); outSize = 16; break;
                        case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI14c_ComplexDot_c7ComplexLj512EEvPvPVKvS4_j"); outSize = 8; break;
                        //case Mode.f_Average_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_Average_ffLj512EEvPvPVKvjj"); outSize = 4; break;
                        //case Mode.c_Average_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11c_Average_c7ComplexLj512EEvPvPVKvjj"); outSize = 8; break;
                        default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                    }
                }
            }

            kernel.DynamicSharedMemory = 512 * outSize;
            kernel.GridDimensions = blockCount;
            kernel.BlockDimensions = 512;
            return kernel;
        }

        /// <summary>
        /// OBSOLETE WILL BE DELETED SOON
        /// </summary>
        public static MyCudaKernel AsyncKernel(int nGPU, Mode mode, int tempBlockSize, int segmentCount = 1, bool segmented = false, bool distributed = false)
        {
            uint outSize = 0;
            MyCudaKernel kernel = null;

            if (segmentCount < 1)
                throw new ArgumentOutOfRangeException("segmentCount", "Invalid block count -- must be positive.");
            if (!segmented && segmentCount > 1) // The global mem barrier in reduction.cu will may create race conditions when called concurrently
                throw new ArgumentOutOfRangeException("segmentCount", "Non-segmented mode called with multiple blocks is not allowed.");

            if (!segmented)
            {
                switch (mode)
                {
                    case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7i_Sum_iiLj512EEvPvPVKvS1_jjjjb"); outSize = 4; break;
                    case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MinIdx_2iiLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11i_MaxIdx_2iiLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvS1_jjjjb"); outSize = 16; break;
                    case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7f_Sum_ffLj512EEvPvPVKvS1_jjjjb"); outSize = 4; break;
                    case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fifLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinIdx_fffLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fifLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MaxIdx_fffLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI11f_MinMax_2ffLj512EEvPvPVKvS1_jjjjb"); outSize = 8; break;
                    case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvS1_jjjjb"); outSize = 16; break;
                    case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7i_Dot_iiLj512EEvPvjPVKvS3_S1_jb"); outSize = 4; break;
                    case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI7f_Dot_ffLj512EEvPvjPVKvS3_S1_jb"); outSize = 4; break;
                    case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI10f_Cosine_ffLj512EEvPvjPVKvS3_S1_jb"); outSize = 16; break;
                    case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10DotProductI14c_ComplexDot_c7ComplexLj512EEvPvjPVKvS4_S2_jb"); outSize = 8; break;
                    case Mode.c_Sum_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z9ReductionI7c_Sum_c7ComplexLj512EEvPvPVKvjjjjb"); outSize = 8; break;
                    default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                }
            }
            else // segmented
            {
                if (!distributed)
                {
                    switch (mode)
                    {
                        case Mode.i_Sum_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7i_Sum_iiLj512EEvPvPVKvS1_jj"); outSize = 4; break;
                        case Mode.i_MinIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MinIdx_2iiLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.i_MaxIdx_2i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11i_MaxIdx_2iiLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.i_MinIdxMaxIdx_4i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvS1_jj"); outSize = 16; break;
                        case Mode.f_Sum_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7f_Sum_ffLj512EEvPvPVKvS1_jj"); outSize = 4; break;
                        case Mode.f_MinIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fifLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.f_MinIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinIdx_fffLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.f_MaxIdx_fi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fifLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.f_MaxIdx_ff: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MaxIdx_fffLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.f_MinMax_2f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_MinMax_2ffLj512EEvPvPVKvS1_jj"); outSize = 8; break;
                        case Mode.f_MinIdxMaxIdx_fifi: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvS1_jj"); outSize = 16; break;
                        case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7i_Dot_iiLj512EEvPvPVKvS3_S1_j"); outSize = 4; break;
                        case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI7f_Dot_ffLj512EEvPvPVKvS3_S1_j"); outSize = 4; break;
                        case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI10f_Cosine_ffLj512EEvPvPVKvS3_S1_j"); outSize = 16; break;
                        case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z11SDotProductI14c_ComplexDot_c7ComplexLj512EEvPvPVKvS4_S2_j"); outSize = 8; break;
                        case Mode.c_Sum_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI7c_Sum_c7ComplexLj512EEvPvPVKvS2_jj"); outSize = 8; break;
                        default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                    }
                }
                else // distributed
                {
                    switch (mode)
                    {
                        case Mode.i_DotProduct_i: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI7i_Dot_iiLj512EEvPvPVKvS3_S1_j"); outSize = 4; break;
                        case Mode.f_DotProduct_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI7f_Dot_ffLj512EEvPvPVKvS3_S1_j"); outSize = 4; break;
                        case Mode.f_Cosine_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI10f_Cosine_ffLj512EEvPvPVKvS3_S1_j"); outSize = 16; break;
                        case Mode.c_ComplexDot_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z12SSDotProductI14c_ComplexDot_c7ComplexLj512EEvPvPVKvS4_S2_j"); outSize = 8; break;
                        //case Mode.f_Average_f: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11f_Average_ffLj512EEvPvPVKvS1_jj"); outSize = 4; break;
                        //case Mode.c_Average_c: kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Reduction\Reduction", "_Z10SReductionI11c_Average_c7ComplexLj512EEvPvPVKvS2_jj"); outSize = 8; break;
                        default: throw new ArgumentOutOfRangeException("mode", "Unrecognized reduction mode.");
                    }
                }
            }

            kernel.DynamicSharedMemory = 512 * outSize;
            kernel.GridDimensions = segmentCount;
            kernel.BlockDimensions = 512;

            if (tempBlockSize < kernel.GridDimensions.x * kernel.GridDimensions.y * kernel.GridDimensions.z * 2)
                throw new ArgumentOutOfRangeException("tempBlockSize", "Temp block size is too small to use with this setting.");

            return kernel;
        }
    }
}
