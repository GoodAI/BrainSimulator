using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CLIWrapper;
using GoodAI.Core.Utils;
using System.Collections.Generic;
using GoodAI.Core.Nodes;
using GoodAI.Modules.Common;
using System.Reflection;

// For debugging:
// Make sure, that your setting is Test->Test setting->Default processor architecture->X64
// SetEntryAssembly is necessary. When calling managed code from unmanaged, GetEntryAssembly may return null (it is called in MyResources)
//  this solves it - code from http://stackoverflow.com/a/21888521

namespace CLIUnitTests
{
    [TestClass]
    public class CLIWrapperTest
    {
        BSCLI CLI;

        /// <summary>
        /// Use as first line in ad hoc tests (needed by XNA specifically)
        /// </summary>
        public static void SetEntryAssembly()
        {
            SetEntryAssembly(Assembly.GetCallingAssembly());
        }

        /// <summary>
        /// Allows setting the Entry Assembly when needed. 
        /// Use AssemblyUtilities.SetEntryAssembly() as first line in XNA ad hoc tests
        /// </summary>
        /// <param name="assembly">Assembly to set as entry assembly</param>
        public static void SetEntryAssembly(Assembly assembly)
        {
            AppDomainManager manager = new AppDomainManager();
            FieldInfo entryAssemblyfield = manager.GetType().GetField("m_entryAssembly", BindingFlags.Instance | BindingFlags.NonPublic);
            entryAssemblyfield.SetValue(manager, assembly);

            AppDomain domain = AppDomain.CurrentDomain;
            FieldInfo domainManagerField = domain.GetType().GetField("_domainManager", BindingFlags.Instance | BindingFlags.NonPublic);
            domainManagerField.SetValue(domain, manager);
        }

        [TestInitialize]
        public void SetupCLI()
        {
            SetEntryAssembly();
            CLI = new BSCLI(MyLogLevel.WARNING);
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\Breakout.brain");
        }

        [TestMethod]
        public void GetNodesOfType()
        {
            List<MyNode> cbs = CLI.GetNodesOfType(typeof(MyRandomNode));
            var test = "ahoj";
            MyLog.DEBUG.WriteLine(test);
        }

        [TestCleanup]
        public void CleanupCLI()
        {
            CLI.Quit();
        }
    }
}
