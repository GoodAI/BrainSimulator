using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Testing;
using Xunit;
using Xunit.Sdk;
using GoodAI.Modules.Common;

namespace BasicNodesTests
{
    public class MyCsvFileWriterNodeTests
    {
        private static readonly string m_outputFileName = string.Format(@"csv_test_{0}.log", Guid.NewGuid().ToString().Substring(0, 10));
        private static readonly string m_outputPath = @"c:\" + m_outputFileName;
        private int m_openCount;

        private AutoResetEvent m_continueEvent = new AutoResetEvent(false);

        /// <summary>
        /// When the simulation is paused or stopped, the handle should be released so that the file can be modified.
        /// </summary>
        [Fact]
        public void FileHandleReleaseTesoutputPathts()
        {
            string directory = Path.GetFullPath(@"Data\");
            const string fileName = "csv_file_test.brain";
            string projectPath = directory + fileName;

            var simulation = new MyLocalSimulation();

            /*MyConfiguration.SetupModuleSearchPath();
            MyConfiguration.LoadModules();*/

            // TODO(HonzaS): This should not be required!
            // The referenced assemblies get loaded only if a Type is required here. But since the serializer
            // is just looking for types by name, it doesn't force the runtime to load all of the assemblies.
            // In this case, we would miss the BasicNodes if the following line was deleted.
            // Two solutions: 1) Use the Managed Extensibility Framework or 2) load all referenced assemblies
            // explicitely (as a part of the BS testing framework).
            var csvNode = new MyCsvFileWriterNode();

            MyProject project;
            using (var reader = new StreamReader(new FileStream(projectPath, FileMode.Open, FileAccess.Read)))
                project = MyProject.Deserialize(reader.ReadToEnd(), Path.GetDirectoryName(projectPath));

            var handler = new MySimulationHandler(simulation)
            {
                Project = project
            };

            // The CSV node
            MyNode node = project.Network.GetChildNodeById(6);
            PropertyInfo property = node.GetType().GetProperty("OutputFile", BindingFlags.Instance | BindingFlags.Public);
            property.SetValue(node, m_outputFileName);

            handler.UpdateMemoryModel();

            handler.StateChanged += StateChanged;

            try
            {
                handler.StartSimulation(oneStepOnly: false);
                m_continueEvent.WaitOne();

                Assert.Throws<IOException>(() =>
                {
                    File.Open(m_outputPath, FileMode.Open, FileAccess.ReadWrite);
                });

                handler.PauseSimulation();
                m_continueEvent.WaitOne();

                handler.StopSimulation();
                m_continueEvent.WaitOne();

                handler.StartSimulation(oneStepOnly: false);
                m_continueEvent.WaitOne();

                handler.StopSimulation();
                m_continueEvent.WaitOne();

                Assert.Equal(3, m_openCount);
            }
            finally
            {
                handler.Finish();
                handler.Dispose();
                File.Delete(m_outputPath);
            }
        }

        private void StateChanged(object sender, MySimulationHandler.StateEventArgs e)
        {
            if (e.NewState == MySimulationHandler.SimulationState.STOPPED || e.NewState == MySimulationHandler.SimulationState.PAUSED)
                OpenFile();
            m_continueEvent.Set();
        }

        private void OpenFile()
        {
            using(var file = File.Open(m_outputPath, FileMode.Open, FileAccess.ReadWrite))
                m_openCount++;
        }
    }
}
