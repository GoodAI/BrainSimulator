using GoodAI.Core;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.RBM;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Matrix;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    /// Feed forward task for a fully connected layer.
    /// </summary>
    /// <description></description>
    [Description("FeedForward"), MyTaskInfo(OneShot = false)]
    public class MyFCForwardTask : MyAbstractForwardTask<MyAbstractWeightLayer>
    {
        public MyFCForwardTask() { } //parameterless constructor

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_forwardBatchKernel;
        private MyCudaKernel m_L1TermKernel;
        private MyCudaKernel m_L2TermKernel;
        private MyCudaKernel m_softmaxKernel;
        public override void Init(int nGPU)
        {
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "FullyConnectedForwardKernel");
            m_forwardBatchKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "FullyConnectedForwardBatchKernel");
            m_L1TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L1TermKernel");
            m_L2TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L2TermKernel");
            m_softmaxKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Activation\ActivationFunction", "SoftmaxKernel");
        }

        public override void Execute() //Task execution
        {
            float dropout = Owner.ParentNetwork.Dropout;

            // skip output layer dropout
            if (Owner is MyOutputLayer)
                dropout = 0;

            if (Owner.ParentNetwork.BatchSize == 1)
            {
                // cuBLAS tends to be slower when BatchSize is 1, use the kernel instead
                m_forwardKernel.SetupExecution(Owner.Neurons);
                m_forwardKernel.Run(
                    (int)Owner.ActivationFunction,
                    Owner.Input,
                    Owner.Output,
                    Owner.Weights,
                    Owner.NeuronInput,
                    Owner.Bias,
                    Owner.DropoutMask,
                    dropout,
                    Owner.Input.Count,
                    Owner.Output.Count
                );
            }
            else
            {
                // NeuronInput = Weights x Input
                MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                    Owner.Neurons, Owner.ParentNetwork.BatchSize, Owner.Input.Count / Owner.ParentNetwork.BatchSize, 1.0f,
                    Owner.Weights.GetDevice(Owner), Owner.Neurons,
                    Owner.Input.GetDevice(Owner), Owner.Input.Count / Owner.ParentNetwork.BatchSize,
                    0.0f, Owner.NeuronInput.GetDevice(Owner), Owner.Neurons
                    );

                // add bias to neuron input and compute activation
                m_forwardBatchKernel.SetupExecution(Owner.Neurons * Owner.ParentNetwork.BatchSize);
                m_forwardBatchKernel.Run(
                    (int)Owner.ActivationFunction,
                    Owner.Output,
                    Owner.NeuronInput,
                    Owner.Bias,
                    Owner.DropoutMask,
                    dropout,
                    Owner.Neurons,
                    Owner.ParentNetwork.BatchSize
                );
            }

            if (Owner.ParentNetwork.L1 > 0) // don't take performance hit if L1 is not used
            {
                m_L1TermKernel.SetupExecution(m_L1TermKernel.MAX_THREADS);
                m_L1TermKernel.DynamicSharedMemory = m_L1TermKernel.BlockDimensions.x * sizeof(float);
                m_L1TermKernel.Run(
                    Owner.Weights,
                    Owner.L1Term,
                    Owner.Weights.Count
                    );
            }

            if (Owner.ParentNetwork.L2 > 0) // don't take performance hit if L2 is not used
            {
                m_L2TermKernel.SetupExecution(m_L2TermKernel.MAX_THREADS);
                m_L2TermKernel.DynamicSharedMemory = m_L2TermKernel.BlockDimensions.x * sizeof(float);
                m_L2TermKernel.Run(
                    Owner.Weights,
                    Owner.L2Term,
                    Owner.Weights.Count
                    );
            }


            // do a trick to avoid infinity (NaN) problems with exponential values in softmax
            if (Owner.ActivationFunction == ActivationFunctionType.SOFTMAX)
            {
                Owner.Output.SafeCopyToHost();

                float expSum = 0;

                float[] f = Owner.Output.Host;
                float max = f.Max();

                for (int i = 0; i < f.Length; i++)
                {
                    float exp = (float) Math.Exp(f[i] - max);
                    f[i] = exp;
                    expSum += exp;
                }


                // CPU version of the commented kernel below
                for (int i = 0; i < f.Length; i++)
                {
                    f[i] /= expSum;
                }

                Array.Copy(f, Owner.Output.Host, f.Length);

                Owner.Output.SafeCopyToDevice();

                /* 
                 * GPU version is slower, don't use it for now
                m_softmaxKernel.SetupExecution(Owner.Neurons);
                m_softmaxKernel.Run(
                    Owner.Output,
                    expSum,
                    Owner.Neurons
                    );*/

            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    /// Backpropagate the deltas to a fully connected previous layer.
    /// </summary>
    /// <description></description>
    [Description("DeltaBack"), MyTaskInfo(OneShot = false)]
    public class MyFCBackDeltaTask : MyAbstractBackDeltaTask<MyAbstractWeightLayer>
    {
        public MyFCBackDeltaTask() { } //parameterless constructor

        private MyCudaKernel m_deltaKernel; // kernel
        private MyCudaKernel m_deltaBatchKernel; // batch kernel
        public override void Init(int nGPU)
        {
            m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "FullyConnectedDeltaKernel");
            m_deltaBatchKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "FullyConnectedDeltaBatchKernel");
        }

        public override void Execute() //Task execution
        {
            MyNode node = Owner.Input.Owner;

            if (node is MyAbstractLayer)
            {
                MyAbstractLayer previousLayer = node as MyAbstractLayer;

                // reset delta only if next is not Gaussian HACK.
                // (Gaussian layer already reseted delta and filled with regularization deltas)
                previousLayer.Delta.Fill(0);

                // determine input to previous layer
                CUdeviceptr prevInputPtr = MyAbstractLayer.DetermineInput(previousLayer);

                if (Owner.ParentNetwork.BatchSize == 1)
                {
                    // cuBLAS tends to be slower when BatchSize is 1, use the kernel instead
                    m_deltaKernel.SetupExecution(previousLayer.Neurons);
                    m_deltaKernel.Run(
                        (int)previousLayer.ActivationFunction,
                        prevInputPtr,
                        previousLayer.Delta,
                        Owner.Delta,
                        Owner.Weights,
                        Owner.ParentNetwork.Dropout,
                        previousLayer.Neurons,
                        Owner.Neurons
                    );
                }
                else
                {
                    // previousLayer.Delta = Transpose(Weights) x Delta
                    MyCublasFactory.Instance.Gemm(Operation.Transpose, Operation.NonTranspose,
                        previousLayer.Neurons, Owner.ParentNetwork.BatchSize, Owner.Neurons, 1.0f,
                        Owner.Weights.GetDevice(Owner), Owner.Neurons,
                        Owner.Delta.GetDevice(Owner), Owner.Neurons,
                        0.0f, previousLayer.Delta.GetDevice(Owner), previousLayer.Neurons
                        );

                    // multiply previousLayer.Delta by activation derivatives of previous layer
                    m_deltaBatchKernel.SetupExecution(previousLayer.Neurons * Owner.ParentNetwork.BatchSize);
                    m_deltaBatchKernel.Run(
                        (int)previousLayer.ActivationFunction,
                        prevInputPtr,
                        previousLayer.Delta,
                        Owner.ParentNetwork.Dropout,
                        previousLayer.Neurons,
                        Owner.ParentNetwork.BatchSize
                    );
                }
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    /// Updates weights, that are fully connected to the previous layer.
    /// </summary>
    /// <description></description>
    [Description("UpdateWeights"), MyTaskInfo(OneShot = false)]
    public class MyFCUpdateWeightsTask : MyAbstractUpdateWeightsTask<MyAbstractWeightLayer>
    {
        public MyFCUpdateWeightsTask() { } //parameterless constructor

        public override void Init(int nGPU) { }

        public override void Execute() //Task execution
        {
            // get enabled loss function
            MyTask task = Owner.ParentNetwork.GetEnabledTask("BackPropagation");
            MyAbstractBackpropTask backpropTask = null;
            if (task is MyAbstractBackpropTask)
                backpropTask = task as MyAbstractBackpropTask;
            else
                MyLog.ERROR.WriteLine("Backprop task does not derive from MyAbstractBackpropTask in " + Owner.ParentNetwork);

            if (backpropTask == null)
                MyLog.ERROR.WriteLine("Undetermined backprop task in " + Owner.ParentNetwork);
            else
            {
                backpropTask.Execute(Owner); // call the group task to do the backpropagation
            }
        }
    }
}
