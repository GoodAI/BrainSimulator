using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Testing;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;
using Rhino.Mocks;
using Xunit;

namespace CoreTests
{
    public class SimulationHandlerTests : IDisposable
    {
        public SimulationHandlerTests()
        {
            TypeMap.InitializeConfiguration<CoreContainerConfiguration>();
        }

        [Fact]
        public void SimulationPropertySetterTest()
        {
            var validator = TypeMap.GetInstance<MyValidator>();
            var simulation = MockRepository.GenerateStub<MySimulation>(validator);
            var handler = new MySimulationHandler(simulation);

            // This should not throw, it's the first simulation.

            Assert.Throws<InvalidOperationException>(() => handler.Simulation = simulation);
        }

        public class TestingNode : MyWorkingNode
        {
            public MySimulationHandler.SimulationState PreviousState = MySimulationHandler.SimulationState.STOPPED;
            public MySimulationHandler.SimulationState CurrentState = MySimulationHandler.SimulationState.STOPPED;
            public AutoResetEvent Event { get; set; }

            [MyInputBlock]
            public MyMemoryBlock<float> InputBlock { get; set; }

            public override void UpdateMemoryBlocks()
            {
            }

            public override void OnSimulationStateChanged(MySimulationHandler.StateEventArgs args)
            {
                PreviousState = args.OldState;
                CurrentState = args.NewState;
                if (Event != null)
                    Event.Set();
            }

            public TestingTask Task { get; set; }
        }

        public class TestingTask : MyTask
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
            }
        }

        // Test that the various state changes get propagated into nodes.
        [Fact]
        public void SimulationStateChangedOnNodesTest()
        {
            var simulation = TypeMap.GetInstance<MySimulation>();
            var handler = new MySimulationHandler(simulation);

            MyProject project = new MyProject
            {
                Network = new MyNetwork()
            };
            project.CreateWorld(typeof(MyTestingWorld));

            var node = project.CreateNode<TestingNode>();
            node.Event = new AutoResetEvent(false);
            project.Network.AddChild(node);
            var connection = new MyConnection(project.Network.GroupInputNodes[0], project.Network.Children[0]);
            connection.Connect();

            project.Network.PrepareConnections();

            handler.Project = project;
            handler.UpdateMemoryModel();

            handler.StartSimulation();
            node.Event.WaitOne();
            Assert.Equal(MySimulationHandler.SimulationState.STOPPED, node.PreviousState);
            Assert.Equal(MySimulationHandler.SimulationState.RUNNING, node.CurrentState);

            handler.PauseSimulation();
            node.Event.WaitOne();
            Assert.Equal(MySimulationHandler.SimulationState.RUNNING, node.PreviousState);
            Assert.Equal(MySimulationHandler.SimulationState.PAUSED, node.CurrentState);

            handler.StartSimulation(stepCount: 1u);
            node.Event.WaitOne();   // Here the sim goes from paused to RUNNING_STEP.
            Assert.Equal(MySimulationHandler.SimulationState.PAUSED, node.PreviousState);
            Assert.Equal(MySimulationHandler.SimulationState.RUNNING_STEP, node.CurrentState);
            node.Event.WaitOne();   // Here it goes to PAUSED.
            Assert.Equal(MySimulationHandler.SimulationState.RUNNING_STEP, node.PreviousState);
            Assert.Equal(MySimulationHandler.SimulationState.PAUSED, node.CurrentState);

            handler.StopSimulation();
            node.Event.WaitOne();
            Assert.Equal(MySimulationHandler.SimulationState.PAUSED, node.PreviousState);
            Assert.Equal(MySimulationHandler.SimulationState.STOPPED, node.CurrentState);

            handler.Finish();
        }

        public void Dispose()
        {
            TypeMap.Destroy();
        }
    }
}
