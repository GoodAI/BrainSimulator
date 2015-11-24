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
        public TestDiscoverer()
        {
            TestBinDirectory = Directory.GetCurrentDirectory();
        }

        public string TestBinDirectory { get; set; }

        public IEnumerable<BrainTest> FindTests()
        {
            var originalLogLevel = MyLog.Level;
            MyLog.Level = MyLogLevel.INFO;

            var testList = new List<BrainTest>();

            foreach (string testAssemblyFullPath in FindTestAssemblies())
            {
                try
                {
                    MyLog.INFO.WriteLine("Searching for tests in '{0}'.", Path.GetFileName(testAssemblyFullPath));

                    Assembly assembly = Assembly.LoadFrom(testAssemblyFullPath);

                    testList.AddRange(assembly.GetTypes()
                        .Where(t => !t.IsAbstract && t.IsSubclassOf(typeof(BrainTest)))
                        .Select(type => (BrainTest)Activator.CreateInstance(type)).ToList());
                }
                catch (Exception e)
                {
                   MyLog.ERROR.WriteLine("Error loading tests from assembly '{0}': {1}", testAssemblyFullPath, e.Message); 
                }
            }

            MyLog.Level = originalLogLevel;

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
    }
}
