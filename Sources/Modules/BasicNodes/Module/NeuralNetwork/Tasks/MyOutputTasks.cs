using GoodAI.Core;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    /// Measures the distance from the target with the commonly used squared loss function.
    /// </summary>
    /// <description></description>
    [Description("SquaredLoss"), MyTaskInfo(OneShot = false)]
    public class MySquaredLossTask : MyAbstractLossTask<MyAbstractOutputLayer>
    {
        public MySquaredLossTask() { } //parameterless constructor

        private MyCudaKernel m_lossKernel; // kernel
        public override void Init(int nGPU)
        {
            m_lossKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\LossFunctions\SquaredLossKernel", "SquaredLossKernel");
        }

        public override void Execute() //Task execution
        {
            // reset delta
            Owner.Delta.Fill(0);

            // get output layer delta
            m_lossKernel.SetupExecution(m_lossKernel.MAX_THREADS);
            m_lossKernel.DynamicSharedMemory = m_lossKernel.BlockDimensions.x * sizeof(float);
            m_lossKernel.Run(
                (int)Owner.ActivationFunction,
                Owner.NeuronInput,
                Owner.Output,
                Owner.Target,
                Owner.Delta,
                Owner.Cost,
                Owner.Neurons,
                Owner.ParentNetwork.BatchSize
                );

            // IMPORTANT: Add regularization
            Owner.AddRegularization();
        }
    }


    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    ///     Standard cross-entropy loss function. Use with output layer with softmax activation function.
    ///     Multiclass target vector with '0's and exactly one '1' is expected.
    /// </summary>
    /// <description></description>
    [Description("CrossEntropyLoss"), MyTaskInfo(OneShot = false)]
    public class MyCrossEntropyLossTask : MyAbstractLossTask<MyAbstractOutputLayer>
    {
        private MyCudaKernel m_lossKernel; // kernel
        public override void Init(int nGPU)
        {
            m_lossKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\LossFunctions\CrossEntropyKernel", "CrossEntropyKernel");
        }

        public override void Execute() //Task execution
        {

            // reset delta
            Owner.Delta.Fill(0);

            // get output layer delta
            m_lossKernel.SetupExecution(m_lossKernel.MAX_THREADS);
            m_lossKernel.DynamicSharedMemory = m_lossKernel.BlockDimensions.x * sizeof(float);
            m_lossKernel.Run(
                (int)Owner.ActivationFunction,
                Owner.NeuronInput,
                Owner.Output,
                Owner.Target,
                Owner.Delta,
                Owner.Cost,
                Owner.Neurons
                );

            // IMPORTANT: Add regularization
            Owner.AddRegularization();
        }
    }

}
