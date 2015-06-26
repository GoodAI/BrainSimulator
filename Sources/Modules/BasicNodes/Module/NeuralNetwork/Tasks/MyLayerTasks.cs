using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Task;
using System.Collections;
using System.ComponentModel;

using YAXLib;
using ManagedCuda;
using BrainSimulator.NeuralNetwork.Layers;

namespace BrainSimulator.NeuralNetwork.Tasks
{
    [Description("InitLayerDeprecated"), MyTaskInfo(OneShot = true)]
    public class MyInitLayerTask : MyTask<MyLayer>
    {
        public MyInitLayerTask() { } //parameterless constructor
        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute() //Task execution
        {
            // init vars to 0
            Owner.PreviousBiasDelta.Fill(0);
            Owner.PreviousWeightDelta.Fill(0);

            // init random weights
            float stdDev = 1.0f / (float)Math.Sqrt(Owner.Input.Count + 1);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Weights.GetDevice(Owner), 0, stdDev);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Bias.GetDevice(Owner), 0, stdDev);
        }
    }

    [Description("FeedForwardDeprecated"), MyTaskInfo(OneShot = false)]
    public class MyFeedForwardTask : MyTask<MyLayer>
    {
        private MyCudaKernel m_kernel;

        public MyFeedForwardTask() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernel", "FeedForwardKernel");
        }

        public override void Execute() //Task execution
        {
            //this will setup thread dimensions (or you can set it on the kernel itself)
            m_kernel.SetupExecution(Owner.Neurons);

            //runs a kernel with given parameters
            m_kernel.Run(
                (int)Owner.ActivationFunction,
                Owner.Input,
                Owner.Output,
                Owner.Weights,
                Owner.WeightedInput,
                Owner.Bias,
                Owner.Input.Count,
                Owner.Output.Count
                );
        }
    }

    [Description("UpdateWeightsDeprecated"), MyTaskInfo(OneShot = false)]
    public class MyUpdateWeightsTaskDeprecated : MyTask<MyLayer>
    {
        private MyCudaKernel m_kernel;

        public MyUpdateWeightsTaskDeprecated() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernel", "FullyConnectedSGDUpdateKernel");
        }

        public override void Execute() //Task execution
        {
            m_kernel.SetupExecution(Owner.Neurons);
            m_kernel.Run(
                Owner.Input,
                Owner.Delta,
                Owner.Weights,
                Owner.PreviousWeightDelta,
                Owner.Bias,
                Owner.PreviousBiasDelta,
                Owner.ParentNetwork.SGD.TrainingRate,
                Owner.ParentNetwork.SGD.Momentum,
                Owner.Input.Count,
                Owner.Neurons
                );
            return;
        }
    }

    [Description("CalcDeltaDeprecated"), MyTaskInfo(OneShot = false)]
    public class MyCalcDeltaTask : MyTask<MyLayer>
    {
        private MyCudaKernel m_kernel;

        public MyCalcDeltaTask() { } //parameterless constructor

        private MyCudaKernel m_outputKernel; //additional kernels

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\HiddenDeltaKernel", "hiddenDeltaKernel");
            m_outputKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\OutputDeltaKernel", "outputDeltaKernel");
        }

        //Task execution
        public override void Execute()
        {
            //// reset delta
            //Owner.Delta.Fill(0); // do this after updating weights (batch learning)

            if (Owner.ProvideTarget)
            {
                // calculate output layer delta
                m_outputKernel.SetupExecution(Owner.Neurons);
                m_outputKernel.Run(
                    (int)Owner.ActivationFunction,
                    Owner.Output,
                    Owner.Target,
                    Owner.Delta,
                    Owner.WeightedInput,
                    Owner.Neurons
                    );
            }
            else
            {
                //// reset delta
                //Owner.Delta.Fill(0); // do this after updating weights (batch learning)

                // calculate hidden layer delta
                MyLayer nextLayer = Owner.NextLayer as MyLayer;
                m_kernel.SetupExecution(Owner.Neurons);
                m_kernel.Run(
                    (int)Owner.ActivationFunction,
                    Owner.WeightedInput,
                    Owner.Delta,
                    nextLayer.Delta,
                    nextLayer.Weights,
                    Owner.Neurons,
                    nextLayer.Neurons
                    );
            }
        }
    }

}