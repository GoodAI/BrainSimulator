using System;
using System.IO;
using System.Linq;
using System.Reflection;
using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.Runtime.InteropServices;
using GoodAI.Core.Memory;

namespace GoodAI.Modules.Transforms
{

    public class MyParallelKernel<T> where T : struct
    {
        protected int m_nGPU;
        protected MyNode m_owner;
        protected MyCudaKernel m_kernel;

        protected MyMemoryBlock<float> m_buffer { get; private set; }

        protected int m_TSize;
        protected int m_outTypeSize;
        protected int m_inTypeSize;

        protected dim3 m_blockDims;
        private dim3 m_gridDims;

        public int timeOffset;
        public int outOffset;
        public int inOffset;

        public dim3 gridDims
        {
            get
            {
                if (m_buffer.Count <= gridDims.x * m_outTypeSize)
                    throw new Exception("Output buffer size is too small (will overflow).");
                return m_gridDims;
            }
            set
            {
                m_gridDims.x = value.x;
                if (value.y != 0 || value.z != 0)
                    MyLog.Writer.WriteLine(MyLogLevel.WARNING, "MyReduction.cs: Attempt to set reduction gridDims with non-zero y and z values.");
                m_kernel.SetupExecution(m_blockDims, m_gridDims);
            }
        }

        protected static ParallelKernelDescriptor GetDescriptor(Type enumType, string enumString)
        {
            MemberInfo[] memInfo = enumType.GetMember(enumString);
            object[] attributes = memInfo[0].GetCustomAttributes(typeof(ParallelKernelDescriptor), false);
            if (attributes.Length == 0)
                throw new Exception("Kernel descriptor is missing for " + enumString + ".");
            return attributes[0] as ParallelKernelDescriptor;
        }

        public MyParallelKernel(MyNode owner, int nGPU, ParallelKernelDescriptor descriptor, int bufferSize)
        {
            m_owner = owner;
            m_nGPU = nGPU;

            m_outTypeSize = descriptor.outTypeSize;
            m_inTypeSize = descriptor.inTypeSize;

            m_TSize = Marshal.SizeOf(typeof(T));
            if (m_TSize > m_inTypeSize || m_TSize > m_outTypeSize || m_inTypeSize % m_TSize != 0 ||
                m_outTypeSize % m_TSize != 0 || m_inTypeSize == 0 || m_outTypeSize == 0)
                MyLog.Writer.WriteLine(MyLogLevel.WARNING, "MyReduction.cs: MemoryBlock type can be incompatible with reduction in/out types.");

            m_kernel = MyKernelFactory.Instance.Kernel(m_nGPU, @"Common\Reduction\Reduction", descriptor.name);

            m_buffer = MyMemoryManager.Instance.CreateMemoryBlock<float>(owner);
            m_buffer.Count = bufferSize;

            m_blockDims = new dim3(512);
            m_gridDims = new dim3(1);

            m_kernel.SetupExecution(m_blockDims, m_gridDims);
            m_kernel.DynamicSharedMemory = 512 * (uint)m_outTypeSize;
        }
    }

    public enum ReductionMode
    {
        [ParallelKernelDescriptor("_Z9ReductionI7i_Sum_iiLj512EEvPvPVKvS1_jjjjb", 4, 4)]
        i_Sum_i,
        [ParallelKernelDescriptor("_Z9ReductionI11i_MinIdx_2iiLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        i_MinIdx_2i,
        [ParallelKernelDescriptor("_Z9ReductionI11i_MaxIdx_2iiLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        i_MaxIdx_2i,
        [ParallelKernelDescriptor("_Z9ReductionI17i_MinIdxMaxIdx_4iiLj512EEvPvPVKvS1_jjjjb", 4, 16)]
        i_MinIdxMaxIdx_4i,
        [ParallelKernelDescriptor("_Z9ReductionI7f_Sum_ffLj512EEvPvPVKvS1_jjjjb", 4, 4)]
        f_Sum_f,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MinMax_2ffLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MinMax_2f,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MinIdx_fifLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MinIdx_fi,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MaxIdx_fifLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MaxIdx_fi,
        [ParallelKernelDescriptor("_Z9ReductionI19f_MinIdxMaxIdx_fififLj512EEvPvPVKvS1_jjjjb", 4, 16)]
        f_MinIdxMaxIdx_fifi,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MinIdx_fffLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MinIdx_ff,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MaxIdx_fffLj512EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MaxIdx_ff,
        [ParallelKernelDescriptor("_Z9ReductionI7c_Sum_c7ComplexLj512EEvPvPVKvS2_jjjjb", 8, 8)]
        c_Sum_c,
    }

    public sealed class MyReductionKernel<T> : MyParallelKernel<T> where T : struct
    {
        public int stride;
        public bool segmented;

        public MyReductionKernel(MyNode owner, int nGPU, ReductionMode mode, bool segmented = false, int bufferSize = 8192)
            : base(owner, nGPU, GetDescriptor(typeof(ReductionMode), mode.ToString()), bufferSize)
        {
            this.stride = 1;
            this.segmented = segmented;
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input)
        {
            CUdeviceptr outputPtr = output.GetDevicePtr(m_nGPU, outOffset * (m_outTypeSize / m_TSize), timeOffset);
            CUdeviceptr inputPtr = input.GetDevicePtr(m_nGPU, inOffset * (m_inTypeSize / m_TSize), timeOffset);

            m_kernel.Run(outputPtr, inputPtr, 0, input.Count, 0, 0, stride, Convert.ToInt32(segmented));
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input)
        {
            CUdeviceptr outputPtr = output.GetDevicePtr(m_nGPU, outOffset * (m_outTypeSize / m_TSize), timeOffset);
            CUdeviceptr inputPtr = input.GetDevicePtr(m_nGPU, inOffset * (m_inTypeSize / m_TSize), timeOffset);

            m_kernel.RunAsync(stream, outputPtr, inputPtr, m_buffer, input.Count, 0, 0, stride, Convert.ToInt32(segmented));
        }
    }

    public enum ProductMode
    {
        [ParallelKernelDescriptor("_Z10DotProductI7f_Dot_ffLj512EEvPvjPVKvS3_S1_jbb", 4, 4)]
        f_DotProduct_f,
        [ParallelKernelDescriptor("_Z10DotProductI7i_Dot_iiLj512EEvPvjPVKvS3_S1_jbb", 4, 4)]
        i_DotProduct_i,
        [ParallelKernelDescriptor("_Z10DotProductI10f_Cosine_ffLj512EEvPvjPVKvS3_S1_jbb", 4, 16)]
        f_Cosine_f,
        [ParallelKernelDescriptor("_Z10DotProductI14c_ComplexDot_c7ComplexLj512EEvPvjPVKvS4_S2_jbb", 8, 8)]
        c_ComplexDot_c
    }

    public class MyProductKernel<T> : MyParallelKernel<T> where T : struct
    {
        public bool segmented;
        public bool distributed;

        public MyProductKernel(MyNode owner, int nGPU, ProductMode mode, bool segmented = false, bool distributed = false, int bufferSize = 8192)
            : base(owner, nGPU, GetDescriptor(typeof(ProductMode), mode.ToString()), bufferSize)
        {
            this.segmented = segmented;
            this.distributed = distributed;
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            CUdeviceptr outputPtr = output.GetDevicePtr(m_nGPU, outOffset * (m_outTypeSize / m_TSize), timeOffset);

            m_kernel.Run(outputPtr, 0, input1, input2, m_buffer, input2.Count, Convert.ToInt32(segmented), Convert.ToInt32(distributed));
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            CUdeviceptr outputPtr = output.GetDevicePtr(m_nGPU, outOffset * (m_outTypeSize / m_TSize), timeOffset);

            m_kernel.RunAsync(stream, outputPtr, 0, input1, input2, m_buffer, input2.Count, Convert.ToInt32(segmented), Convert.ToInt32(distributed));
        }
    }

    public class ParallelKernelDescriptor : System.Attribute
    {
        public string name;
        public int inTypeSize;
        public int outTypeSize;

        public ParallelKernelDescriptor(string name, int inTypeSize, int outTypeSize)
        {
            this.name = name;
            this.inTypeSize = inTypeSize;
            this.outTypeSize = outTypeSize;
        }
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
