using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
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
            var testList = new List<BrainTest>();

            foreach (string testAssemblyFullPath in FindTestAssemblies())
            {
                try
                {
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

            return testList;
        }

        private IEnumerable<string> FindTestAssemblies()
        {
            if (string.IsNullOrEmpty(TestBinDirectory))
                return new List<string>();

            return Directory.GetFiles(TestBinDirectory, @"*BrainTests.dll");
        }
    }
}
