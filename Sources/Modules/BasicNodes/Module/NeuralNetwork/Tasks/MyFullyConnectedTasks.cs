using BrainSimulator;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Layers;
using BrainSimulator.RBM;
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
    [Description("FeedForward"), MyTaskInfo(OneShot = false)]
    public class MyFCForwardTask : MyAbstractForwardTask<MyAbstractWeightLayer>
    {
        public MyFCForwardTask() { } //parameterless constructor

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_L1TermKernel;
        private MyCudaKernel m_L2TermKernel;
        public override void Init(int nGPU)
        {
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "FullyConnectedForwardKernel");
            m_L1TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L1TermKernel");
            m_L2TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L2TermKernel");
        }

        public override void Execute() //Task execution
        {
            float dropout = Owner.ParentNetwork.Dropout;

            // skip output layer dropout
            if (Owner is MyOutputLayer)
                dropout = 0;

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
        }
    }

    [Description("DeltaBack"), MyTaskInfo(OneShot = false)]
    public class MyFCBackDeltaTask : MyAbstractBackDeltaTask<MyAbstractWeightLayer>
    {
        public MyFCBackDeltaTask() { } //parameterless constructor

        private MyCudaKernel m_deltaKernel; // kernel
        public override void Init(int nGPU)
        {
            m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "FullyConnectedDeltaKernel");
        }

        public override void Execute() //Task execution
        {
            // pointer to previous layer
            MyAbstractLayer previousLayer = Owner.PreviousLayer;

            if (previousLayer != null)
            {
                // reset delta
                previousLayer.Delta.Fill(0);

                // determine input to previous layer
                CUdeviceptr prevInputPtr;
                if (previousLayer is MyAbstractWeightLayer)
                    prevInputPtr = (previousLayer as MyAbstractWeightLayer).NeuronInput.GetDevicePtr(previousLayer.GPU);
                else
                    prevInputPtr = previousLayer.Input.GetDevicePtr(previousLayer.GPU);

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
        }
    }

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
                backpropTask.Execute(Owner); // call the group task to do the backpropagation
        }
    }
}
