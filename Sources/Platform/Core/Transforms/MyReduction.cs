using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

namespace GoodAI.Modules.Transforms
{
    public enum ThreadCount:int { T32 = 32,  T64 = 64,  T128 = 128, T256 = 256, T512 = 512 }

    public class ParallelKernelDescriptor : System.Attribute
    {
        public string nameFirstHalf;
        public string nameSecondHalf;
        public int inTypeSize;
        public int outTypeSize;
        public string modeName;

        public string GetKernelName(ThreadCount threads)
        {
            return nameFirstHalf + ((int)threads).ToString() + nameSecondHalf;
        }

        public static ParallelKernelDescriptor GetDescriptor(Type type, string name)
        {
            MemberInfo[] memInfo = type.GetMember(name);
            object[] attributes = memInfo[0].GetCustomAttributes(typeof(ParallelKernelDescriptor), false);
            if (attributes.Length == 0)
                throw new Exception("Kernel descriptor is missing for " + name + ".");
            ParallelKernelDescriptor descriptor = attributes[0] as ParallelKernelDescriptor;
            descriptor.modeName = name;
            return descriptor;
        }

        public ParallelKernelDescriptor(string nameFirstHalf, string nameSecondHalf, int inTypeSize, int outTypeSize)
        {
            this.nameFirstHalf = nameFirstHalf;
            this.nameSecondHalf = nameSecondHalf;
            this.inTypeSize = inTypeSize;
            this.outTypeSize = outTypeSize;
        }
    }

    public class MyParallelKernel<T> where T : struct
    {
        protected int m_nGPU;
        protected MyNode m_owner;

        protected Dictionary<ThreadCount, MyCudaKernel> m_kernels;

        public const int BUFFER_SIZE = 8192;
        protected MyMemoryBlock<float> m_buffer { get; private set; }

        protected int m_TSize;
        protected int m_outTypeSize;
        protected int m_inTypeSize;

        protected dim3 m_gridDims;
        protected dim3 m_blockDims;

        public int size;
        public int outOffset;
        public int inOffset;
        public bool segmented;

        private ThreadCount m_threadCount;
        public ThreadCount threadCount
        {
            get { return m_threadCount; }
            set
            {
                m_blockDims.x = (uint)((int)m_threadCount);
                m_kernels[m_threadCount].SetupExecution(m_blockDims, m_gridDims);
                m_kernels[m_threadCount].DynamicSharedMemory = (uint)((int)m_threadCount * m_outTypeSize);
            }
        }

        public int segments
        {
            get
            {
                if (segmented && m_buffer.Count <= (outOffset + m_gridDims.x) * m_outTypeSize ||
                    m_buffer.Count <= (outOffset + 1) * m_outTypeSize)
                    throw new Exception("Output buffer size is too small (will overflow).");
                return (int)m_gridDims.x;
            }
            set
            {
                if (value < 1) throw new Exception("Grid dimension has to be positive.");
                m_gridDims.x = (uint)value;
                m_kernels[m_threadCount].SetupExecution(m_blockDims, m_gridDims);
                m_kernels[m_threadCount].DynamicSharedMemory = (uint)((int)m_threadCount * m_outTypeSize);
            }
        }

        protected virtual void ResetParameters()
        {
            m_threadCount = ThreadCount.T32;
            size = outOffset = inOffset = 0;
            segments = 1;
            segmented = false;
        }

        public MyParallelKernel(MyNode owner, int nGPU, ParallelKernelDescriptor descriptor, int bufferSize)
        {
            m_owner = owner;
            m_nGPU = nGPU;

            m_threadCount = ThreadCount.T32;

            m_outTypeSize = descriptor.outTypeSize;
            m_inTypeSize = descriptor.inTypeSize;

            m_TSize = Marshal.SizeOf(typeof(T));
            if (m_TSize > m_inTypeSize || m_TSize > m_outTypeSize || m_inTypeSize % m_TSize != 0 ||
                m_outTypeSize % m_TSize != 0 || m_inTypeSize == 0 || m_outTypeSize == 0)
                MyLog.Writer.WriteLine(MyLogLevel.WARNING, "MyReduction.cs: MemoryBlock type can be incompatible with reduction in/out types.");

            Array threadCountValues = Enum.GetValues(typeof(ThreadCount));
            m_kernels = new Dictionary<ThreadCount, MyCudaKernel>();
            foreach (ThreadCount threadCount in Enum.GetValues(typeof(ThreadCount)))
            {
                string kernelName = descriptor.GetKernelName(threadCount);
                m_kernels.Add(threadCount, MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\Reduction\Reduction", kernelName));
            }

            m_buffer = MyMemoryManager.Instance.CreateMemoryBlock<float>(owner);
            m_buffer.Name = "Buffer(" + descriptor.modeName + ")";
            m_buffer.Count = bufferSize;

            m_blockDims = new dim3((int)m_threadCount, 1, 1);
            m_gridDims = new dim3(1, 1, 1);
        }
    }

    public enum ReductionMode
    {
        [ParallelKernelDescriptor("_Z9ReductionI7i_Sum_iiLj", "EEvPvPVKvS1_jjjjb", 4, 4)]
        i_Sum_i,
        [ParallelKernelDescriptor("_Z9ReductionI11i_MinIdx_2iiLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        i_MinIdx_2i,
        [ParallelKernelDescriptor("_Z9ReductionI11i_MaxIdx_2iiLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        i_MaxIdx_2i,
        [ParallelKernelDescriptor("_Z9ReductionI17i_MinIdxMaxIdx_4iiLj", "EEvPvPVKvS1_jjjjb", 4, 16)]
        i_MinIdxMaxIdx_4i,
        [ParallelKernelDescriptor("_Z9ReductionI7f_Sum_ffLj", "EEvPvPVKvS1_jjjjb", 4, 4)]
        f_Sum_f,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MinMax_2ffLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MinMax_2f,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MinIdx_fifLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MinIdx_fi,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MaxIdx_fifLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MaxIdx_fi,
        [ParallelKernelDescriptor("_Z9ReductionI19f_MinIdxMaxIdx_fififLj", "EEvPvPVKvS1_jjjjb", 4, 16)]
        f_MinIdxMaxIdx_fifi,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MinIdx_fffLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MinIdx_ff,
        [ParallelKernelDescriptor("_Z9ReductionI11f_MaxIdx_fffLj", "EEvPvPVKvS1_jjjjb", 4, 8)]
        f_MaxIdx_ff,
        [ParallelKernelDescriptor("_Z9ReductionI7c_Sum_c7ComplexLj", "EEvPvPVKvS2_jjjjb", 8, 8)]
        c_Sum_c,
    }

    public class MyReductionKernel<T> : MyParallelKernel<T> where T : struct
    {
        public int stride;

        protected override void ResetParameters()
        {
            base.ResetParameters();

            stride = 1;
        }

        public MyReductionKernel(MyNode owner, int nGPU, ReductionMode mode, int bufferSize = BUFFER_SIZE)
            : base(owner, nGPU, ParallelKernelDescriptor.GetDescriptor(typeof(ReductionMode), mode.ToString()), bufferSize)
        {
            ResetParameters();
        }

        private void KernelRun(CudaStream stream, CUdeviceptr outputPtr, CUdeviceptr inputPtr, int size, bool async = false)
        {
            CUdeviceptr bufferPtr = m_buffer.GetDevicePtr(m_nGPU);
            outputPtr = outputPtr + outOffset * m_outTypeSize;
            inputPtr = inputPtr + inOffset * m_inTypeSize;

            if (this.size > 0)
                size = this.size;

            int segmentedFlag = Convert.ToInt32(segmented);

            if (async)
            {
                m_kernels[threadCount].RunAsync(stream, outputPtr, inputPtr, bufferPtr, size, outOffset, inOffset, stride, Convert.ToInt32(segmented));
            }
            else
            {
                m_kernels[threadCount].Run(outputPtr, inputPtr, bufferPtr, size, outOffset, inOffset, stride, Convert.ToInt32(segmented));
            }
            
            ResetParameters();
        }

        /* OVERLOADS - SYNCHRONOUS */

        public void Run(CUdeviceptr outputPtr, CUdeviceptr inputPtr, int size)
        {
            KernelRun(null, outputPtr, inputPtr, size);
        }

        public void Run(MyMemoryBlock<T> output, CUdeviceptr inputPtr, int size)
        {
            KernelRun(null, output.GetDevicePtr(m_nGPU), inputPtr, size);
        }

        public void Run(CUdeviceptr outputPtr, MyMemoryBlock<T> input)
        {
            KernelRun(null, outputPtr, input.GetDevicePtr(m_nGPU), input.Count);
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input)
        {
            KernelRun(null, output.GetDevicePtr(m_nGPU), input.GetDevicePtr(m_nGPU), input.Count);
        }

        /* OVERLOADS - ASYNCHRONOUS */

        public void RunAsync(CudaStream stream, CUdeviceptr outputPtr, CUdeviceptr inputPtr, int size)
        {
            KernelRun(stream, outputPtr, inputPtr, size, true);
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, CUdeviceptr inputPtr, int size)
        {
            KernelRun(stream, output.GetDevicePtr(m_nGPU), inputPtr, size, true);
        }

        public void RunAsync(CudaStream stream, CUdeviceptr outputPtr, MyMemoryBlock<T> input)
        {
            KernelRun(stream, outputPtr, input.GetDevicePtr(m_nGPU), input.Count, true);
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input)
        {
            KernelRun(stream, output.GetDevicePtr(m_nGPU), input.GetDevicePtr(m_nGPU), input.Count, true);
        }
    }

    public enum ProductMode
    {
        [ParallelKernelDescriptor("_Z10DotProductI7f_Dot_ffLj", "EEvPvjPVKvS3_S1_jbb", 4, 4)]
        f_DotProduct_f,
        [ParallelKernelDescriptor("_Z10DotProductI7i_Dot_iiLj", "EEvPvjPVKvS3_S1_jbb", 4, 4)]
        i_DotProduct_i,
        [ParallelKernelDescriptor("_Z10DotProductI10f_Cosine_ffLj", "EEvPvjPVKvS3_S1_jbb", 4, 16)]
        f_Cosine_f,
        [ParallelKernelDescriptor("_Z10DotProductI14c_ComplexDot_c7ComplexLj", "EEvPvjPVKvS4_S2_jbb", 8, 8)]
        c_ComplexDot_c
    }

    public class MyProductKernel<T> : MyParallelKernel<T> where T : struct
    {
        public bool distributed;

        protected override void ResetParameters()
        {
            base.ResetParameters();

            distributed = false;
        }

        public MyProductKernel(MyNode owner, int nGPU, ProductMode mode, int bufferSize = BUFFER_SIZE)
            : base(owner, nGPU, ParallelKernelDescriptor.GetDescriptor(typeof(ProductMode), mode.ToString()), bufferSize)
        {
            ResetParameters();
        }

        private void KernelRun(CudaStream stream, CUdeviceptr outputPtr, CUdeviceptr input1Ptr, CUdeviceptr input2Ptr, int size, bool async = false)
        {
            CUdeviceptr bufferPtr = m_buffer.GetDevicePtr(m_nGPU);
            outputPtr = outputPtr + outOffset * m_outTypeSize;

            if (this.size > 0)
                size = this.size;

            int segmentedFlag = Convert.ToInt32(segmented);
            int distributedFlag = Convert.ToInt32(distributed);

            if (async)
            {
                m_kernels[threadCount].RunAsync(stream, outputPtr, outOffset, input1Ptr, input2Ptr, bufferPtr, size, segmentedFlag, distributedFlag);
            }
            else
            {
                m_kernels[threadCount].Run(outputPtr, outOffset, input1Ptr, input2Ptr, bufferPtr, size, segmentedFlag, distributedFlag);
            }
            
            ResetParameters();
        }

        /* OVERLOADS - SYNCHRONOUS */

        public void Run(CUdeviceptr outputPtr, CUdeviceptr input1Ptr, CUdeviceptr input2Ptr, int size)
        {
            KernelRun(null, outputPtr, input1Ptr, input2Ptr, size);
        }

        public void Run(CUdeviceptr outputPtr, MyMemoryBlock<T> input1, CUdeviceptr input2Ptr)
        {
            KernelRun(null, outputPtr, input1.GetDevicePtr(m_nGPU), input2Ptr, input1.Count);
        }

        public void Run(CUdeviceptr outputPtr, CUdeviceptr input1Ptr, MyMemoryBlock<T> input2, int size)
        {
            KernelRun(null, outputPtr, input1Ptr, input2.GetDevicePtr(m_nGPU), size);
        }

        public void Run(CUdeviceptr outputPtr, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            KernelRun(null, outputPtr, input1.GetDevicePtr(m_nGPU), input2.GetDevicePtr(m_nGPU), input1.Count);
        }

        public void Run(MyMemoryBlock<T> output, CUdeviceptr input1Ptr, CUdeviceptr input2Ptr, int size)
        {
            KernelRun(null, output.GetDevicePtr(m_nGPU), input1Ptr, input2Ptr, size);
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input1, CUdeviceptr input2Ptr)
        {
            KernelRun(null, output.GetDevicePtr(m_nGPU), input1.GetDevicePtr(m_nGPU), input2Ptr, input1.Count);
        }

        public void Run(MyMemoryBlock<T> output, CUdeviceptr input1Ptr, MyMemoryBlock<T> input2, int size)
        {
            KernelRun(null, output.GetDevicePtr(m_nGPU), input1Ptr, input2.GetDevicePtr(m_nGPU), size);
        }

        public void Run(MyMemoryBlock<T> output, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            KernelRun(null, output.GetDevicePtr(m_nGPU), input1.GetDevicePtr(m_nGPU), input2.GetDevicePtr(m_nGPU), input1.Count);
        }

        /* OVERLOADS - ASYNCHRONOUS */

        public void RunAsync(CudaStream stream, CUdeviceptr outputPtr, CUdeviceptr input1Ptr, CUdeviceptr input2Ptr, int size)
        {
            KernelRun(stream, outputPtr, input1Ptr, input2Ptr, size, true);
        }

        public void RunAsync(CudaStream stream, CUdeviceptr outputPtr, MyMemoryBlock<T> input1, CUdeviceptr input2Ptr)
        {
            KernelRun(stream, outputPtr, input1.GetDevicePtr(m_nGPU), input2Ptr, input1.Count, true);
        }

        public void RunAsync(CudaStream stream, CUdeviceptr outputPtr, CUdeviceptr input1Ptr, MyMemoryBlock<T> input2, int size)
        {
            KernelRun(stream, outputPtr, input1Ptr, input2.GetDevicePtr(m_nGPU), size, true);
        }

        public void RunAsync(CudaStream stream, CUdeviceptr outputPtr, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            KernelRun(stream, outputPtr, input1.GetDevicePtr(m_nGPU), input2.GetDevicePtr(m_nGPU), input1.Count, true);
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, CUdeviceptr input1Ptr, CUdeviceptr input2Ptr, int size)
        {
            KernelRun(stream, output.GetDevicePtr(m_nGPU), input1Ptr, input2Ptr, size, true);
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input1, CUdeviceptr input2Ptr)
        {
            KernelRun(stream, output.GetDevicePtr(m_nGPU), input1.GetDevicePtr(m_nGPU), input2Ptr, input1.Count, true);
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, CUdeviceptr input1Ptr, MyMemoryBlock<T> input2, int size)
        {
            KernelRun(stream, output.GetDevicePtr(m_nGPU), input1Ptr, input2.GetDevicePtr(m_nGPU), size, true);
        }

        public void RunAsync(CudaStream stream, MyMemoryBlock<T> output, MyMemoryBlock<T> input1, MyMemoryBlock<T> input2)
        {
            KernelRun(stream, output.GetDevicePtr(m_nGPU), input1.GetDevicePtr(m_nGPU), input2.GetDevicePtr(m_nGPU), input1.Count, true);
        }
    }
}
