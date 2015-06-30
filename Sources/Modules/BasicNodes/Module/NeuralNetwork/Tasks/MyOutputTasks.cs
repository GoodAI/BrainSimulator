using BrainSimulator;
using BrainSimulator.NeuralNetwork.Layers;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.NeuralNetwork.Tasks
{
    /// <author>Philip Hilm</author>
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
                Owner.Neurons
                );

            // IMPORTANT: Add regularization
            Owner.AddRegularization();
        }
    }

    [Description("CrossEntropy"), MyTaskInfo(OneShot = false)]
    public class MyCrossEntropyDeltaTask : MyAbstractLossTask<MyAbstractOutputLayer>
    {
        public MyCrossEntropyDeltaTask() { } //parameterless constructor

        private MyCudaKernel m_lossDeltaKernel; // kernel
        public override void Init(int nGPU)
        {
            m_lossDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\LossFunction\CrossEntropyDeltaKernel", "CrossEntropyDeltaKernel");
        }

        public override void Execute() //Task execution
        {
            // reset delta
            Owner.Delta.Fill(0); // do this after updating weights (batch learning)

            // get output layer delta
            m_lossDeltaKernel.SetupExecution(Owner.Neurons);
            m_lossDeltaKernel.Run(
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
