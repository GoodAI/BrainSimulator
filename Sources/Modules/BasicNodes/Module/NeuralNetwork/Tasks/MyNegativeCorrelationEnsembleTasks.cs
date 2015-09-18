using GoodAI.Core;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.RBM;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Core.Memory;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Core.Nodes;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>mbr</meta>
    /// <status>Working</status>
    /// <summary>
    /// Initialize Negative Correlation.
    /// </summary>
    /// <description></description>
    [Description("Init"), MyTaskInfo(OneShot = true)]
    public class MyNegativeCorrelationInitTask : MyTask<MyNegativeCorrelationEnsembleLayer>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.Delta.Fill(0);
            Owner.Output.Fill(0);
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mbr</meta>
    /// <status>Working</status>
    /// <summary>
    /// Tasks for NegativeCorrelation output layer.
    /// </summary>
    /// <description></description>
    [Description("FeedForward"), MyTaskInfo(OneShot = false)]
    public class MyNegativeCorrelationForwardTask : MyAbstractForwardTask<MyNegativeCorrelationEnsembleLayer>
    {
        Random rnd = new Random();

        private MyCudaKernel m_forwardResetKernel;
        private MyCudaKernel m_forwardSumKernel;
        private MyCudaKernel m_forwardDivideKernel;

        public MyNegativeCorrelationForwardTask() { }

        public override void Init(int nGPU)
        {
            m_forwardResetKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "NegativeCorrelationForwardResetKernel");
            m_forwardResetKernel.SetupExecution(Owner.Neurons);
            
            m_forwardSumKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "NegativeCorrelationForwardSumKernel");
            m_forwardSumKernel.SetupExecution(Owner.Neurons);
            
            m_forwardDivideKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "NegativeCorrelationForwardDivideKernel");
            m_forwardDivideKernel.SetupExecution(Owner.Neurons);
        }

        public override void Execute()
        {
            // Input.Count == Output.Count (it is validated)
            // Sum kernels + Divide kernel = Average
            // First sum elementwise all input into output
            m_forwardResetKernel.Run(
                Owner.Output,
                Owner.Output.Count
            );

            foreach (MyConnection connection in Owner.InputConnections)
            {
                if (connection.From is MyAbstractLayer)
                {
                    MyAbstractLayer prevLayer = connection.From as MyAbstractLayer;

                    m_forwardSumKernel.Run(
                        prevLayer.Output,
                        Owner.Output,
                        Owner.Output.Count
                    );
                }
            }

            // Then divide output by input size
            m_forwardDivideKernel.Run(
                Owner.Output,
                Owner.Output.Count,
                Owner.InputBranches
            );
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mbr</meta>
    /// <status>Working</status>
    /// <summary>
    /// Backpropagate Negative Correlation.
    /// </summary>
    /// <description></description>
    [Description("DeltaBack"), MyTaskInfo(OneShot = false)]
    public class MyNegativeCorrelationBackDeltaTask : MyAbstractBackDeltaTask<MyNegativeCorrelationEnsembleLayer>
    {
        [YAXSerializableField(DefaultValue = 0.8f)]
        [MyBrowsable, Category("Hyperparameters")]
        public float Lambda { get; set; }

        private MyCudaKernel m_deltaKernel;

        public MyNegativeCorrelationBackDeltaTask() { }

        public override void Init(int nGPU)
        {
            m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "NegativeCorrelationDeltaKernel");
            m_deltaKernel.SetupExecution(Owner.Neurons);
        }

        public override void Execute()
        {
            Owner.Delta.Fill(0.0f);
            // number of neurons of ensemble is the same as for each input
            m_deltaKernel.SetConstantVariable<float>("Lambda", Lambda);

            int inputLayerCount = Owner.InputConnections.Count(x => x.From is MyAbstractWeightLayer);

            foreach (MyConnection connection in Owner.InputConnections)
            {
                if (connection.From is MyAbstractLayer)
                {
                    MyAbstractLayer prevLayer = connection.From as MyAbstractLayer;

                    if (prevLayer is MyAbstractWeightLayer)
                    {
                        MyAbstractWeightLayer prevWeightLayer = prevLayer as MyAbstractWeightLayer;

                        m_deltaKernel.Run(
                            (int)prevLayer.ActivationFunction,
                            prevWeightLayer.NeuronInput,
                            prevLayer.Output,
                            Owner.Output,
                            Owner.Neurons,
                            prevLayer.Delta,
                            Owner.Delta,
                            inputLayerCount
                        );
                    }
                    prevLayer.Delta.SafeCopyToHost();
                    Owner.Delta.SafeCopyToHost();
                }
            }
            Owner.Delta.SafeCopyToHost();
        }
    }
}
