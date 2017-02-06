using CoreTests;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.Scripting;
using GoodAI.Modules.Testing;
using GoodAI.Modules.Transforms;
using MNIST;
using Xunit;

namespace BasicNodesTests
{
    public class ModelCreationTests : CoreTestBase
    {
        [Fact]
        public void SimulationRunsViaRunner()
        {
            using (var runner = new MyProjectRunner())
            {
                MyProject project = runner.CreateProject(typeof(MyTestingWorld));

                var node = project.CreateNode<MyCSharpNode>();
                project.Network.AddChild(node);

                runner.RunAndPause(1);
            }
        }

        [Fact]
        public void CreatesAndRunsMNIST()
        {
            using (var runner = new MyProjectRunner())
            {
                MyProject project = runner.CreateProject(typeof(MNISTWorld));

                var world = project.World as MNISTWorld;

                var neuralGroup = project.CreateNode<MyNeuralNetworkGroup>();
                project.Network.AddChild(neuralGroup);

                var hiddenLayer = project.CreateNode<MyHiddenLayer>();
                neuralGroup.AddChild(hiddenLayer);

                var outputLayer = project.CreateNode<MyOutputLayer>();
                neuralGroup.AddChild(outputLayer);

                var accumulator = project.CreateNode<MyAccumulator>();
                neuralGroup.AddChild(accumulator);

                // Connect the nodes.

                project.Connect(project.Network.GroupInputNodes[0], neuralGroup, 0, 0);
                project.Connect(project.Network.GroupInputNodes[1], neuralGroup, 0, 1);

                project.Connect(neuralGroup.GroupInputNodes[0], hiddenLayer, 0, 0);

                project.Connect(neuralGroup.GroupInputNodes[1], outputLayer, 0, 1);

                project.Connect(hiddenLayer, outputLayer, 0, 0);

                project.Connect(outputLayer, accumulator, 1, 0);

                // Setup the nodes.

                var sendMnistData = world.SendMNISTTrainData;
                Assert.NotNull(sendMnistData);
                sendMnistData.ExpositionTime = 1;

                world.BitmapOrder = ExampleOrderOption.Shuffle;
                world.OneHot = true;

                hiddenLayer.Neurons = 40;

                accumulator.ApproachValue.ApproachMethod = MyAccumulator.MyApproachValueTask.SequenceType.Momentum;
                accumulator.ApproachValue.Delta = 0.1f;
                accumulator.ApproachValue.Target = 0;
                accumulator.ApproachValue.Factor = 0.9f;

                // Enable tasks.
                project.World.EnableDefaultTasks();

                neuralGroup.EnableDefaultTasks();
                neuralGroup.RMS.Enabled = true;

                hiddenLayer.EnableDefaultTasks();

                outputLayer.EnableDefaultTasks();

                accumulator.EnableDefaultTasks();
                accumulator.ApproachValue.Enabled = true;

                // Run the simulation.

                runner.RunAndPause(100);

                float error = runner.GetValues(accumulator.Id)[0];
                Assert.True(error < 0.5f);
                //runner.SaveProject(@"c:\foobar.brain");
            }
        }
    }
}
