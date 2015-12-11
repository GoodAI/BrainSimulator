using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Scripting;
using GoodAI.Modules.Testing;
using Xunit;

namespace CoreTests
{
    public class ModelCreationTests
    {
        public MySimulationHandler GetLocalSimulationHandler()
        {
            return new MySimulationHandler(new MyLocalSimulation());
        }

        public MyProject GetNewProject(Type worldType)
        {
            var project = new MyProject
            {
                Network = new MyNetwork()
            };
            project.CreateWorld(worldType);

            return project;
        }

        [Fact]
        public void SimulationRuns()
        {
            var handler = GetLocalSimulationHandler();
            var project = GetNewProject(typeof(MyTestingWorld));

            var node = project.CreateNode<MyCSharpNode>();
            project.Network.AddChild(node);

            var connection = new MyConnection(project.Network.GroupInputNodes[0], project.Network.Children[0]);
            connection.Connect();

            project.Network.PrepareConnections();

            handler.Project = project;
            handler.UpdateMemoryModel();

            handler.StartSimulation(stepCount: 1);

            handler.Finish();
        }
    }
}
