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
                Owner.Neurons,
                Owner.ParentNetwork.BatchSize
            );

            // IMPORTANT: Add regularization
            Owner.AddRegularization();
        }
    }

    /// <author>GoodAI</author>
    /// <status>Working</status>
    /// <summary>
    ///     Loss function for comparing two images. 
    /// </summary>
    /// <description></description>
    [Description("ImageLoss"), MyTaskInfo(OneShot = false)]
    public class MyImageLossTask : MyAbstractLossTask<MyAbstractOutputLayer>
    {
        private MyCudaKernel m_lossKernel; // kernel
        public override void Init(int nGPU)
        {
            m_lossKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\LossFunctions\ImageLoss", "ImageLossKernel");
        }

        public override void Execute() //Task execution
        {
            // reset delta
            Owner.Delta.Fill(0);

            // get output layer delta
            m_lossKernel.SetupExecution(m_lossKernel.MAX_THREADS);
            m_lossKernel.DynamicSharedMemory = m_lossKernel.BlockDimensions.x * sizeof(float);
            m_lossKernel.Run(
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

    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>
    ///     To minimise your own loss function set Target to derivatives of the loss function w.r.t. network's output.
    ///     A simple example of custom loss function may be loss(output) = output, which has derivative w.r.t. output equal to 1.
    ///     If you set Target to 1, the network tries to minimise its output. Analogously, setting Target to -1 will attempt to maximise output.
    /// </summary>
    /// <description></description>
    [Description("CustomLoss"), MyTaskInfo(OneShot = false)]
    public class MyCustomLossTask : MyAbstractLossTask<MyAbstractOutputLayer>
    {
        private MyCudaKernel m_lossKernel; // kernel
        public override void Init(int nGPU)
        {
            m_lossKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\LossFunctions\CustomLossKernel", "CustomLossKernel");
            m_lossKernel.SetupExecution(Owner.Neurons * Owner.ParentNetwork.BatchSize);
        }

        public override void Execute() //Task execution
        {
            // get output layer delta
            m_lossKernel.Run(
                (int)Owner.ActivationFunction,
                Owner.NeuronInput,
                Owner.Target,
                Owner.Delta,
                Owner.Neurons,
                Owner.ParentNetwork.BatchSize
            );
        }
    }
}
