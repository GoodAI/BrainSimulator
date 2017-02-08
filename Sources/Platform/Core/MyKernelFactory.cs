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
        public int MAX_THREADS { get; protected set; }

        protected CudaKernel m_kernel;
        protected int m_GPU;

        private CudaStream m_stream;

        public string KernelName { get { return m_kernel.KernelName; } }
        public dim3 BlockDimensions { get { return m_kernel.BlockDimensions; } set { m_kernel.BlockDimensions = value; } }
        public dim3 GridDimensions { get { return m_kernel.GridDimensions; } set { m_kernel.GridDimensions = value; } }
        public uint DynamicSharedMemory { get { return m_kernel.DynamicSharedMemory; } set { m_kernel.DynamicSharedMemory = value; } }

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

        public MyCudaKernel(CudaKernel kernel, int GPU, CudaStream stream)
        {
            m_GPU = GPU;
            m_stream = stream;

            m_kernel = kernel;
            MAX_THREADS = m_kernel.MaxThreadsPerBlock;
        }

        public MyCudaKernel(string kernelName, CUmodule module, CudaContext cuda, int GPU, CudaStream stream = null)
            : this(new CudaKernel(kernelName, module, cuda), GPU, stream)
        { }

        /// <summary>
        /// Runs the kernel asynchronously when a non-null CudaStream was injected via the constructor
        /// or synchronously when it was not.
        /// </summary>
        /// <param name="args">MyMemoryBlock arguments are automatically converted to device pointers.</param>
        public void Run(params object[] args)
        {
            if (m_stream != null)
                RunAsync(m_stream, args);
            else
                RunSync(args);
        }

        /// <summary>Runs the kernel in synchronous mode.</summary>
        /// <param name="args">MyMemoryBlock arguments are automatically converted to device pointers.</param>
        public void RunSync(params object[] args)
        {
            ConvertMemoryBlocksToDevicePtrs(args);

            m_kernel.Run(args);
        }

        /// <summary> Runs the kernel in asynchronous mode. </summary>
        /// <param name="stream">If the stream is null, the default per-thread stream is used.</param>
        /// <param name="args">MyMemoryBlock arguments are automatically converted to device pointers.</param>
        public void RunAsync(CudaStream stream, params object[] args)
        {
            ConvertMemoryBlocksToDevicePtrs(args);

            CUstream cuStream = stream?.Stream ?? CUstream.StreamPerThread;

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

        private void ConvertMemoryBlocksToDevicePtrs(object[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (!(args[i] is MyAbstractMemoryBlock))
                    continue;

                args[i] = ((MyAbstractMemoryBlock)args[i]).GetDevicePtr(m_GPU);
                if (((CUdeviceptr)args[i]).Pointer == 0)
                {
                    throw new InvalidOperationException(
                        "Memory block resolved to null device ptr (not allocated on device?).");
                }
            }
        }
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

        private readonly Dictionary<string, CudaKernel>[] m_ptxModules;

        private int m_devCount; // number of CUDA-enabled devices
        private CudaContext[] m_contexts;
        private CudaRandDevice[] m_randDevices;
        private CudaStream[] m_streams;
        private bool[] m_contextAlive;

        public CudaRandDevice GetRandDevice(MyNode callee)
        {
            return m_randDevices[callee.GPU];
        }

        protected MyKernelFactory()
        {
            ContextsCreate();
            m_ptxModules = new Dictionary<string, CudaKernel>[DevCount];

            for (int i = 0; i < DevCount; i++)
            {
                m_ptxModules[i] = new Dictionary<string, CudaKernel>();
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

        private CudaKernel LoadPtxWithLinker(int GPU, string ptxFileName, string kernelName, string additionalLinkDependencyPath)
        {
            var options = new CudaJitOptionCollection();
            var err = new CudaJOErrorLogBuffer(1024);
            options.Add(new CudaJOLogVerbose(true));
            options.Add(err);

            try
            {
                CudaLinker linker = new CudaLinker(options);
                linker.AddFile(ptxFileName, CUJITInputType.PTX, null);
                // Add the requested additional library
                linker.AddFile(additionalLinkDependencyPath, CUJITInputType.Library, null);
                byte[] cubin = linker.Complete();

                return m_contexts[GPU].LoadKernelPTX(cubin, kernelName);
            }
            catch (Exception e)
            {
                throw new CudaException($"CUDA JIT linker error {err.Value}", e);
            }
        }

        private CudaKernel LoadPtx(int GPU, string ptxFileName, string kernelName)
        {
            CUmodule ptxModule = m_contexts[GPU].LoadModule(ptxFileName);
            return new CudaKernel(kernelName, ptxModule, m_contexts[GPU]);
        }

        private MyCudaKernel TryLoadPtx(int GPU, string ptxFileName, string kernelName, bool forceNewInstance, string additionalLinkDependencyPath, bool isFallback = false)
        {
            if (m_ptxModules[GPU].ContainsKey(ptxFileName) && !forceNewInstance)
                return new MyCudaKernel(m_ptxModules[GPU][ptxFileName], GPU, m_streams[GPU]);

            try
            {
                CudaKernel kernel = null;

                if (additionalLinkDependencyPath == null)
                    kernel = LoadPtx(GPU, ptxFileName, kernelName);
                else
                    kernel = LoadPtxWithLinker(GPU, ptxFileName, kernelName, additionalLinkDependencyPath);

                m_ptxModules[GPU][ptxFileName] = kernel;
                return new MyCudaKernel(kernel, GPU, m_streams[GPU]);
            }
            catch (Exception e)
            {
                if (!isFallback && additionalLinkDependencyPath == null)
                {
                    // Simple loading failed, try extended linkage
                    MyLog.WARNING.WriteLine("Kernel loading failed. Fallback to extended linkage...");
                    return KernelInternal(GPU, ptxFileName, kernelName, forceNewInstance, extendedLinkage: true, isFallback: true);
                }
                // Fallback to simple wouldn't make much sense, so we don't do it..

                throw new CudaException($"{e.Message} ({ptxFileName})", e);
            }
        }

        private string GetCudaLibPath(string fileName)
        {
            if (File.Exists(fileName))
                return Path.GetFullPath(fileName);

            MyLog.INFO.WriteLine($"Trying to access a kernel with an extended linkage (which requires the {fileName} library), but could not locate the {fileName} library. Trying CUDA toolkit path.");

            // Try searching in the cuda toolkit, if it is installed
            var cudaPath = Environment.GetEnvironmentVariable(@"CUDA_PATH");

            if (cudaPath == null || !Directory.Exists(cudaPath))
            {
                MyLog.WARNING.WriteLine("Could not locate the CUDA toolkit, because the CUDA_PATH environment variable is not defined or the content is invalid. Please re-install the CUDA toolkit.");
                return null;
            }

            string libPath = Path.Combine(cudaPath, "lib", "x64", fileName);

            if (!File.Exists(libPath))
            {
                libPath = Path.Combine(cudaPath, "bin", fileName);

                if (!File.Exists(libPath))
                    throw new CudaException($"Cannot locate the {fileName} library in cuda toolkit.");
            }

            return libPath;
        }

        private MyCudaKernel KernelInternal(int GPU, string ptxFileName, string kernelName, bool forceNewInstance, bool extendedLinkage, bool isFallback = false)
        {
            string cudaRtPath = null;

            if (extendedLinkage)
            {
                cudaRtPath = GetCudaLibPath("cudadevrt.lib");

                if (cudaRtPath == null)
                {
                    // Lib loading failed
                    if (isFallback)
                        // Don't fallback to anything, if this is already a fallback
                        throw new CudaException($"Failed to load ptx: {ptxFileName}");

                    // Try to omit the extended linkage
                    MyLog.WARNING.WriteLine("Kernel loading failed. Fallback to simple loading...");
                    isFallback = true;
                }
            }

            return TryLoadPtx(GPU, ptxFileName, kernelName, forceNewInstance, cudaRtPath, isFallback);
        }

        public MyCudaKernel Kernel(int GPU, string ptxFolder, string ptxFileName, string kernelName, bool forceNewInstance = false, bool extendedLinkage = false)
        {
            string ptxPath = Path.Combine(ptxFolder, $"{ptxFileName}.ptx");

            if (!File.Exists(ptxPath))
            {
                ptxPath = Path.Combine(MyConfiguration.GlobalPTXFolder, $"{ptxFileName}.ptx");

                if (!File.Exists(ptxPath))
                    throw new CudaException("Cannot find ptx: " + ptxFileName);
            }

            return KernelInternal(GPU, ptxPath, kernelName, forceNewInstance, extendedLinkage);
        }

        private MyCudaKernel Kernel(int GPU, Assembly callingAssembly, string ptxFileName, string kernelName, bool forceNewInstance = false, bool extendedLinkage = false)
        {
            FileInfo assemblyFile = GetAssemblyFile(callingAssembly);
            string ptxFolder = assemblyFile.DirectoryName + @"\ptx\";

            return Kernel(GPU, ptxFolder, ptxFileName, kernelName, forceNewInstance, extendedLinkage);
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
        public MyCudaKernel Kernel(int nGPU, string ptxFileName, string kernelName, bool forceNewInstance = false, bool extendedLinkage = false)
        {
            return Kernel(nGPU, Assembly.GetCallingAssembly(), ptxFileName, kernelName, forceNewInstance, extendedLinkage);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public MyCudaKernel Kernel(int nGPU, string ptxFileName, bool forceNewInstance = false, bool extendedLinkage = false)
        {
            return Kernel(nGPU, Assembly.GetCallingAssembly(), ptxFileName, GetKernelNameFromPtx(ptxFileName), forceNewInstance, extendedLinkage);
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

        public MyCudaKernel KernelVector(int nGPU, KernelVector kernelName)
        {
            //Because the method Kernel is called from this method, it will look (via the Assembly.GetCallingAssembly()) for the kernels in BasicNodesCuda and not in the place from where the method KernelVector is called.
            return Instance.Kernel(nGPU, @"Transforms\KernelVector", kernelName.ToString());
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
            m_streams = new CudaStream[m_devCount];
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

            m_streams[GPU] = new CudaStream(CUstream.StreamPerThread);

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
                    m_streams[i].Dispose();
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
