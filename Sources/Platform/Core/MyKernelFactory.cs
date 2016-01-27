using GoodAI.Core.Configuration;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaRand;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace GoodAI.Core
{
    public class MyCudaKernel
    {
        public int MAX_THREADS { get; protected set;  }

        protected CudaKernel m_kernel;
        protected int m_GPU;

        public string KernelName { get { return m_kernel.KernelName; } }
        public dim3 BlockDimensions { get { return m_kernel.BlockDimensions; } set {m_kernel.BlockDimensions = value; } }
        public dim3 GridDimensions { get { return m_kernel.GridDimensions; } set {m_kernel.GridDimensions = value; } }
        public uint DynamicSharedMemory { get {return m_kernel.DynamicSharedMemory;} set {m_kernel.DynamicSharedMemory = value; } }

        // TODO, there are 109 overloaded versions of this :(
        // now listing only those we use
        public void SetConstantVariable(string name, int value) { m_kernel.SetConstantVariable(name, value); } 
        public void SetConstantVariable(string name, int[] value) { m_kernel.SetConstantVariable(name, value); } 
        public void SetConstantVariable(string name, uint value) { m_kernel.SetConstantVariable(name, value); }
        public void SetConstantVariable(string name, uint[] value) { m_kernel.SetConstantVariable(name, value); }
        public void SetConstantVariable(string name, float value) { m_kernel.SetConstantVariable(name, value); } 
        public void SetConstantVariable(string name, float[] value) { m_kernel.SetConstantVariable(name, value); }
        public void SetConstantVariable(string name, double value) { m_kernel.SetConstantVariable(name, value); }

        // this one will be tough to move to grid
        public void SetConstantVariable<T>(string name, T value) where T : struct { m_kernel.SetConstantVariable<T>(name, value); }
        //public void SetConstantVariable(string name, CUdeviceptr value) { m_kernel.SetConstantVariable(name, value); } 

        public MyCudaKernel(string kernelName, CUmodule module, CudaContext cuda, int GPU)
        {
            m_GPU = GPU;
            m_kernel = new CudaKernel(kernelName, module, cuda);
            MAX_THREADS = m_kernel.MaxThreadsPerBlock;
        }

        public void Run(params object[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] is MyAbstractMemoryBlock)
                {
                    args[i] = (args[i] as MyAbstractMemoryBlock).GetDevicePtr(m_GPU);
                    if (((CUdeviceptr)args[i]).Pointer == 0)
                    {
                        // TODO(Premek): this is now handled in observers, should be also handled in the simulation
                        throw new InvalidOperationException("Memory block resolved to null device ptr (not allocated on device?).");
                    }
                }
            }

            m_kernel.Run(args);
        }

        public void RunAsync(CudaStream stream, params object[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] is MyAbstractMemoryBlock)
                {
                    args[i] = (args[i] as MyAbstractMemoryBlock).GetDevicePtr(m_GPU);
                    if (((CUdeviceptr)args[i]).Pointer == 0)
                    {
                        // TODO(Premek): this is now handled in observers, should be also handled in the simulation
                        throw new InvalidOperationException("Memory block resolved to null device ptr (not allocated on device?).");
                    }
                }
            }

            CUstream cuStream = CUstream.NullStream;
            if (stream != null)
            {
                cuStream = stream.Stream;
            }
            m_kernel.RunAsync(cuStream, args);
        }

        public void SetupExecution(int numOfParallelUnits)
        {
            if (numOfParallelUnits > MAX_THREADS)
            {
                m_kernel.BlockDimensions = new dim3(MAX_THREADS, 1, 1);
                m_kernel.GridDimensions = new dim3(numOfParallelUnits / MAX_THREADS + 1, 1, 1);
            }
            else
            {
                m_kernel.BlockDimensions = new dim3(numOfParallelUnits, 1, 1);
                m_kernel.GridDimensions = new dim3(1, 1, 1);
            }
        }

        public virtual void SetupExecution(dim3 blockDimensions, dim3 gridDimensions)
        {
            m_kernel.BlockDimensions = blockDimensions;
            m_kernel.GridDimensions = gridDimensions;
        }
    }

    public enum MyKernelMode
    {
        Immediate,
        Scripted
    }

    public class MyKernelFactory : IDisposable
    {

        #region Singleton machinery
        private static MyKernelFactory SINGLETON;

        public static MyKernelFactory Instance
        {
            get
            {
                if (SINGLETON == null)
                {
                    SINGLETON = new MyKernelFactory();
                }
                return SINGLETON;
            }
        }
        #endregion

        // Track whether Dispose has been called. 
        private Boolean m_disposed = false;

        private Dictionary<string, CUmodule>[] m_ptxModules;

        private int m_devCount; // number of CUDA-enabled devices
        private CudaContext[] m_contexts;
        private CudaRandDevice[] m_randDevices;
        private bool[] m_contextAlive;

        public CudaRandDevice GetRandDevice(MyNode callee)
        {
            return m_randDevices[callee.GPU];
        }

        private MyKernelFactory()
        {
            ContextsCreate();
            m_ptxModules = new Dictionary<string, CUmodule>[DevCount];

            for (int i = 0; i < DevCount; i++)
            {
                m_ptxModules[i] = new Dictionary<string, CUmodule>();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (m_disposed)
                return;

            if (disposing)
            {
                ContextsDispose();
            }

            m_disposed = true;
        }

        private MyCudaKernel TryLoadPtx(int GPU, string ptxFileName, string kernelName, bool forceNewInstance = false)
        {
            if (m_ptxModules[GPU].ContainsKey(ptxFileName) && !forceNewInstance)
            {
                return new MyCudaKernel(kernelName, m_ptxModules[GPU][ptxFileName], m_contexts[GPU], GPU);
            }
            else
            {
                try
                {
                    FileInfo ptxFile = new FileInfo(ptxFileName);

                    if (ptxFile.Exists)
                    {
                        CUmodule ptxModule = m_contexts[GPU].LoadModule(ptxFileName);
                        m_ptxModules[GPU][ptxFileName] = ptxModule;

                        return new MyCudaKernel(kernelName, m_ptxModules[GPU][ptxFileName], m_contexts[GPU], GPU);
                    }
                    else return null;
                }
                catch (Exception e)
                {
                    throw new CudaException(e.Message + " (" + ptxFileName + ")", e);
                }
            }
        }

        public MyCudaKernel Kernel(int GPU, string ptxFolder, string ptxFileName, string kernelName, bool forceNewInstance = false)
        {
            MyCudaKernel kernel = TryLoadPtx(GPU, ptxFolder + ptxFileName + ".ptx", kernelName, forceNewInstance);

            if (kernel == null)
            {                
                kernel = TryLoadPtx(GPU, MyConfiguration.GlobalPTXFolder + ptxFileName + ".ptx", kernelName, forceNewInstance);
            }

            if (kernel == null)
            {
                throw new CudaException("Cannot find ptx: " + ptxFileName);
            }

            return kernel;
        }

        private MyCudaKernel Kernel(int GPU, Assembly callingAssembly, string ptxFileName, string kernelName, bool forceNewInstance = false)
        {
            FileInfo assemblyFile = GetAssemblyFile(callingAssembly);
            string ptxFolder = assemblyFile.DirectoryName + @"\ptx\";

            return Kernel(GPU, ptxFolder, ptxFileName, kernelName, forceNewInstance);
        }

        private static FileInfo GetAssemblyFile(Assembly callingAssembly)
        {
            return MyConfiguration.AssemblyLookup[callingAssembly.FullName].File;
        }

        private static string GetKernelNameFromPtx(string ptxFileName)
        {
            return ptxFileName.Substring(ptxFileName.LastIndexOf('\\') + 1);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public MyCudaKernel Kernel(int nGPU, string ptxFileName, string kernelName, bool forceNewInstance = false)
        {
            return Kernel(nGPU, Assembly.GetCallingAssembly(), ptxFileName, kernelName, forceNewInstance);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public MyCudaKernel Kernel(int nGPU, string ptxFileName, bool forceNewInstance = false)
        {
            return Kernel(nGPU, Assembly.GetCallingAssembly(), ptxFileName, GetKernelNameFromPtx(ptxFileName), forceNewInstance);            
        }

        public MyReductionKernel<T> KernelReduction<T>(MyNode owner, int nGPU, ReductionMode mode,
            int bufferSize = MyParallelKernel<T>.BUFFER_SIZE, bool forceNewInstance = false) where T : struct
        {
            return new MyReductionKernel<T>(owner, nGPU, mode, bufferSize);
        }

        public MyProductKernel<T> KernelProduct<T>(MyNode owner, int nGPU, ProductMode mode,
            int bufferSize = MyParallelKernel<T>.BUFFER_SIZE, bool forceNewInstance = false) where T : struct
        {
            return new MyProductKernel<T>(owner, nGPU, mode, bufferSize);
        }

        // !!! Warning: This is for testing purposes only.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public MyCudaKernel Kernel(string name, bool forceNewInstance = false)
        {
            return Kernel(DevCount - 1, Assembly.GetCallingAssembly(), name, GetKernelNameFromPtx(name), forceNewInstance);                        
        }


        public int DevCount
        {
            get { return m_devCount; }
        }

        public CudaContext GetContextByGPU(int nGPU)
        {
            Debug.Assert(nGPU >= 0 && nGPU < DevCount, "Bad GPU ID.");
            return m_contexts[nGPU];
        }

        public List<Tuple<SizeT, SizeT>> GetMemInfo()
        {
            SizeT free = 0, total = 0;
            List<Tuple<SizeT, SizeT>> result = new List<Tuple<SizeT, SizeT>>();

            for (int i = 0; i < DevCount; i++)
            {
                SetCurrent(i);
                ManagedCuda.DriverAPINativeMethods.MemoryManagement.cuMemGetInfo_v2(ref free, ref total);
                result.Add(new Tuple<SizeT, SizeT>(free, total));
            }

            return result;
        }

        public void SetCurrent(int nGPU)
        {
            m_contexts[nGPU].SetCurrent();
        }

        /**
         * Creates all CUDA contexts
         */
        private void ContextsCreate()
        {
            m_devCount = CudaContext.GetDeviceCount();

            m_contexts = new CudaContext[m_devCount];
            m_randDevices = new CudaRandDevice[m_devCount];
            m_contextAlive = new bool[m_devCount];

            for (int i = 0; i < DevCount; i++)
            {
                CreateContext(i);
            }
        }

        private void CreateContext(int GPU)
        {
            m_contexts[GPU] = new CudaContext(GPU);

            m_contexts[GPU].SetCurrent();
            m_randDevices[GPU] = new CudaRandDevice(GeneratorType.PseudoDefault);
            m_randDevices[GPU].SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks.GetHashCode());

            m_contextAlive[GPU] = true;
        }

        internal void MarkContextDead(int GPU)
        {
            m_contextAlive[GPU] = false;
        }

        internal bool IsContextAlive(int GPU)
        {
            return m_contextAlive[GPU];
        }

        internal void RecoverContexts()
        {
            for (int i = 0; i < DevCount; i++)
            {
                if (!IsContextAlive(i))
                {
                    m_randDevices[i].Dispose();
                    m_ptxModules[i].Clear();

                    m_contexts[i].Dispose();
                    CreateContext(i);

                    MyLog.WARNING.WriteLine("Dead context detected. Restart of BrainSimulator is needed.");
                }
            }
        }

        /**
         * Destroys all CUDA contexts
         */
        private void ContextsDispose()
        {
            foreach (CudaContext context in m_contexts)
            {
                context.Dispose();
            }
            m_contexts = null;
        }

        public void ClearLoadedKernels()
        {
            for (int i = 0; i < DevCount; i++)
            {
                m_ptxModules[i].Clear();
            }
        }

        public void Synchronize()
        {
            foreach (CudaContext context in m_contexts)
            {
                context.Synchronize();
            }
        }
    }
}
