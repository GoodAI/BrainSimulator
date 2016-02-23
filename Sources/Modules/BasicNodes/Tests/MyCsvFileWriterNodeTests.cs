using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using CoreTests;
using GoodAI.Core;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Testing;
using Xunit;
using Xunit.Sdk;
using GoodAI.Modules.Common;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;

namespace BasicNodesTests
{
    public class MyCsvFileWriterNodeTests : CoreTestBase
    {
        private int m_openCount;

        private readonly AutoResetEvent m_continueEvent = new AutoResetEvent(false);
        private string m_outputFileFullPath;
        private MyValidator m_validator;

        public MyCsvFileWriterNodeTests()
        {
            m_validator = TypeMap.GetInstance<MyValidator>();
        }

        /// <summary>
        /// When the simulation is paused or stopped, the handle should be released so that the file can be modified.
        /// </summary>
        [Fact]
        public void FileCanBeReadWhenSimulationIsNotRunning()
        {
            string directory = Path.GetFullPath(@"Data\");
            const string fileName = "csv_file_test.brain";

            m_outputFileFullPath = Path.GetTempFileName();
            var outputFileName = Path.GetFileName(m_outputFileFullPath);
            var outputFileDirectory = Path.GetDirectoryName(m_outputFileFullPath);

            string projectPath = directory + fileName;

            var simulation = TypeMap.GetInstance<MySimulation>();

            // TODO(HonzaS): This should not be required!
            // The referenced assemblies get loaded only if a Type is required here. But since the serializer
            // is just looking for types by name, it doesn't force the runtime to load all of the assemblies.
            // In this case, we would miss the BasicNodes if the following line was deleted.
            // Two solutions: 1) Use the Managed Extensibility Framework or 2) load all referenced assemblies
            // explicitely (as a part of the BS testing framework).
            var csvNode = new MyCsvFileWriterNode();

            MyProject project = MyProject.Deserialize(File.ReadAllText(projectPath), Path.GetDirectoryName(projectPath));

            var handler = new MySimulationHandler(simulation)
            {
                Project = project
            };

            // The CSV node
            MyNode node = project.Network.GetChildNodeById(6);

            PropertyInfo fileNameProperty = node.GetType().GetProperty("OutputFile", BindingFlags.Instance | BindingFlags.Public);
            fileNameProperty.SetValue(node, outputFileName);

            PropertyInfo directoryProperty = node.GetType().GetProperty("OutputDirectory", BindingFlags.Instance | BindingFlags.Public);
            directoryProperty.SetValue(node, outputFileDirectory);

            handler.UpdateMemoryModel();

            handler.StateChanged += StateChanged;

            try
            {
                handler.StartSimulation();
                m_continueEvent.WaitOne();

                Assert.Throws<IOException>(() =>
                {
                    File.Open(m_outputFileFullPath, FileMode.Open, FileAccess.ReadWrite);
                });

                // Every time we change simulation state, the StateChanged method gets notified.
                // We're changing the state several times here and the StateChanged methods checks that
                // the file that the Csv node uses is available for writing.

                // First, go through start->pause->stop.
                handler.PauseSimulation();
                m_continueEvent.WaitOne();

                handler.StopSimulation();
                m_continueEvent.WaitOne();

                // Now, try start->stop only.
                handler.StartSimulation();
                m_continueEvent.WaitOne();

                handler.StopSimulation();
                m_continueEvent.WaitOne();

                // The file should have been successfully opened three times - every time the simulation was paused or stopped.
                Assert.Equal(3, m_openCount);
            }
            finally
            {
                handler.Finish();
                File.Delete(m_outputFileFullPath);
                m_outputFileFullPath = null;
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
            using(var file = File.Open(m_outputFileFullPath, FileMode.Open, FileAccess.ReadWrite))
                m_openCount++;
        }
    }
}
