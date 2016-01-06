using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Configuration;
using GoodAI.Core.Utils;
using GoodAI.Testing.BrainUnit;

namespace GoodAI.Tests.BrainTestRunner
{
    internal class TestDiscoverer
    {
        private readonly string m_filter;

        public TestDiscoverer(string filter)
        {
            m_filter = filter != null ? filter.ToLower() : null;
            TestBinDirectory = Directory.GetCurrentDirectory();

            BrainUnitNodeTestsDirectory =
                Path.Combine(Directory.GetCurrentDirectory(), @"../../../BrainTests/BrainUnitNodeTests/");
        }

        public string TestBinDirectory { get; set; }

        public string BrainUnitNodeTestsDirectory { get; set; }

        public IEnumerable<BrainTest> FindTests()
        {
            MyLogLevel originalLogLevel = MyLog.Level;
            MyLog.Level = MyLogLevel.INFO;

            var testList = new List<BrainTest>();

            try
            {
                testList.AddRange(SafeCollectTests(CollectBinaryTests));
                testList.AddRange(SafeCollectTests(CollectBrainUnitNodeTests));
            }
            finally
            {
                MyLog.Level = originalLogLevel;
            }

            return string.IsNullOrEmpty(m_filter)
                ? testList
                : testList.Where(test => test.Name.ToLower().Contains(m_filter));
        }

        /// <summary>
        /// Safe means "does not throw exceptions".
        /// </summary>
        private static IEnumerable<BrainTest> SafeCollectTests(Func<IEnumerable<BrainTest>> collectTestsFunc)
        {
            try
            {
                return collectTestsFunc();
            }
            catch (Exception)
            {
                MyLog.ERROR.WriteLine("Failed to collect tests by '{0}'.", collectTestsFunc.Method.Name);
                return new List<BrainTest>();
            }
        }

        private IEnumerable<BrainTest> CollectBinaryTests()
        {
            var testList = new List<BrainTest>();

            foreach (string testAssemblyFullPath in FindTestAssemblies())
            {
                try
                {
                    MyLog.INFO.WriteLine("Searching for tests in '{0}'.", Path.GetFileName(testAssemblyFullPath));

                    Assembly assembly = Assembly.LoadFrom(testAssemblyFullPath);

                    testList.AddRange(assembly.GetTypes()
                        .Where(t => !t.IsAbstract && t.IsSubclassOf(typeof (BrainTest)))
                        .Select(type => (BrainTest) Activator.CreateInstance(type)).ToList());
                }
                catch (Exception e)
                {
                    MyLog.ERROR.WriteLine("Error loading tests from assembly '{0}': {1}", testAssemblyFullPath, e.Message);
                }
            }

            return testList;
        }

        private IEnumerable<string> FindTestAssemblies()
        {
            List<string> fileList = FindBrainTestAssemblies().ToList();

            fileList.AddRange(MyConfiguration.ListModules().Select(fileInfo => fileInfo.FullName));

            return fileList;
        }

        private IEnumerable<string> FindBrainTestAssemblies()
        {
            if (string.IsNullOrEmpty(TestBinDirectory))
                return new List<string>();

            return Directory.GetFiles(TestBinDirectory, @"*BrainTests.dll");
        }

        private IEnumerable<BrainTest> CollectBrainUnitNodeTests()
        {
            return FindBrainUnitNodeProjects().Select(path => new BrainUnitNodeTest(path));
        }

        private IEnumerable<string> FindBrainUnitNodeProjects()
        {
            if (string.IsNullOrEmpty(BrainUnitNodeTestsDirectory))
                return new List<string>();

            return Directory.GetFiles(BrainUnitNodeTestsDirectory, @"*.brain?"); 
        }
    }
}
