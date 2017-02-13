using System;
using System.IO;
using GoodAI.Core;
using GoodAI.Core.Utils;
using ManagedCuda;
using Xunit;
using Xunit.Abstractions;

namespace CoreTests
{
    public abstract class KernelFactoryTestBase
    {
        #region Nested helper classes

        protected class TempFileLoader
            : IDisposable
        {
            public string TempFolderName { get; }
            public string TargetPath { get; }

            public TempFileLoader(string filePath)
            {
                TempFolderName = Path.GetRandomFileName();
                Directory.CreateDirectory(TempFolderName);
                TargetPath = Path.Combine(TempFolderName, Path.GetFileName(filePath));

                File.Copy(filePath, TargetPath);
            }

            public void Dispose()
            {
                File.Delete(TargetPath);
                Directory.Delete(TempFolderName);
            }
        }

        private class DebugLogWriter
            : MyLogWriter
        {
            private readonly ITestOutputHelper m_output;

            public DebugLogWriter(ITestOutputHelper output)
            {
                m_output = output;
            }

            public void WriteLine(MyLogLevel level, string message)
            {
                m_output.WriteLine(message + '\n');
            }

            public void Write(MyLogLevel level, string message)
            {
                m_output.WriteLine(message);
            }

            public void Write(MyLogLevel level, char message)
            {
                m_output.WriteLine(message.ToString());
            }

            public void FlushCache()
            { }
        }

        #endregion

        #region Fields and consts

        // Ptx names
        private readonly string m_ptxBase = Path.Combine(Directory.GetCurrentDirectory(), "Data", "Ptx");

        protected const string BasicPtxName = "BasicTest";
        protected const string DynParaPtxName = "DynParaTest";

        private const string EntryName = "Test";

        // Cuda names
        private const string CudaVar = "CUDA_PATH";

        protected static string CudaPath
        {
            get { return Environment.GetEnvironmentVariable(CudaVar); }
            set { Environment.SetEnvironmentVariable(CudaVar, value); }
        }

        protected string RtLibPath => Path.Combine($"{CudaPath}", "lib", "x64", "cudadevrt.lib");

        #endregion

        #region Genesis

        public KernelFactoryTestBase(ITestOutputHelper output)
        {
            //MyLog.Writer = new DebugLogWriter(output);
        }

        #endregion

        #region Helpers

        protected MyCudaKernel GetKernel(string ptxName, bool extendedLinkage)
        {
            return MyKernelFactory.Instance.Kernel(0, m_ptxBase, ptxName, EntryName, true, extendedLinkage);
        }

        #endregion

        #region Simple loading

        [Fact]
        public void LoadBasicSimple()
        {
            // Non-dynamic kernel, simple loading
            var k = GetKernel(BasicPtxName, false);
            Assert.NotNull(k);
        }

        [Fact]
        public void LoadBasicExtended()
        {
            // Non-dynamic kernel, extended linking
            var k = GetKernel(BasicPtxName, true);
            Assert.NotNull(k);
        }

        [Fact]
        public virtual void LoadDynamicExtended()
        {
            // Dynamic kernel, extended linking
            var k = GetKernel(DynParaPtxName, true);
            Assert.NotNull(k);
        }

        [Fact]
        public virtual void LoadDynamicBasic()
        {
            // Dynamic kernel, basic loading
            // The loading should fall back to extended linkage and should not fail
            var k = GetKernel(DynParaPtxName, false);
            Assert.NotNull(k);
        }

        #endregion
    }

    // Tests for gathering the cudadevrt.lib from CUDA tools during extended linking
    public class KernelFactoryToolsLibTests
        : KernelFactoryTestBase
    {
        public KernelFactoryToolsLibTests(ITestOutputHelper output)
            : base(output)
        { }
    }

    // Tests for gathering the cudadevrt.lib from working dir during extended linking
    public class KernelFactoryLocalLibTests
        : KernelFactoryTestBase, IDisposable
    {
        private string m_libFolderBackup;
        private TempFileLoader m_libLoader;

        public KernelFactoryLocalLibTests(ITestOutputHelper output)
            : base(output)
        {
            m_libLoader = new TempFileLoader(RtLibPath);
            m_libFolderBackup = MyKernelFactory.Instance.ExtendedLinkageLibFolder;
            MyKernelFactory.Instance.ExtendedLinkageLibFolder = m_libLoader.TempFolderName;
        }

        public void Dispose()
        {
            MyKernelFactory.Instance.ExtendedLinkageLibFolder = m_libFolderBackup;
            m_libFolderBackup = null;
            m_libLoader.Dispose();
            m_libLoader = null;
        }
    }

    // Tests for extended linking with the cudadevrt.lib missing (from working dir and with invalid cuda path)
    public class KernelFactoryNoLibTests
        : KernelFactoryTestBase, IDisposable
    {
        private readonly string m_oldCudaPath;

        public KernelFactoryNoLibTests(ITestOutputHelper output)
            : base(output)
        {
            m_oldCudaPath = CudaPath;
            CudaPath = "__INVALID_PATH"; // Making the cuda path invalid causes the lib not to be found
        }

        public void Dispose()
        {
            CudaPath = m_oldCudaPath;
        }


        #region Test overrides

        public override void LoadDynamicExtended()
        {
            Assert.Throws<CudaException>(() => GetKernel(DynParaPtxName, true));
        }

        public override void LoadDynamicBasic()
        {
            Assert.Throws<CudaException>(() => GetKernel(DynParaPtxName, false));
        }

        #endregion
    }
}
