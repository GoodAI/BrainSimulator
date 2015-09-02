using CustomModels.RBM.Tasks;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.RBM.Tasks;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.RBM
{
    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    ///     Input layer of Restricted Boltzmann Machine network.
    ///     Can only act as a visible layer. In an RBM network, there is only one input layer and it is its first node.
    /// </summary>
    /// <description>
    /// Index of this layer is (or rather must be) 0.
    /// 
    /// There must be precisely one input layer in an RBM group.
    /// </description>
    public class MyRBMInputLayer : MyAbstractLayer, IMyCustomTaskFactory
    {
        public override ConnectionType Connection
        {
            get { return ConnectionType.ONE_TO_ONE; } // phil inserted to remove warning about connection not set
        }

        public MyRBMInitLayerTask RBMInitLayerTask { get; private set; }

        [ReadOnly(true)]
        public override int Neurons { get; set; }


        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tLayer")]
        public bool ApplyBiases { get; set; }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tLayer")]
        public bool StoreEnergy { get; set; }

        #region Memory blocks

        [MyPersistable]
        public MyMemoryBlock<float> Bias { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> PreviousBiasDelta { get; protected set; }
        public MyMemoryBlock<float> PreviousOutput { get; protected set; }
        public MyMemoryBlock<float> Random { get; protected set; }
        public MyMemoryBlock<float> Energy { get; protected set; }
        #endregion

        public void Init(int nGPU)
        {
            m_RBMInputForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMInputForwardKernel");
            m_RBMInputForwardAndStoreKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMInputForwardAndStoreKernel");
            m_RBMUpdateBiasesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMUpdateBiasesKernel");
            m_RBMRandomActivationKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMRandomActivationKernel");
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
                Neurons = Input.Count;

            base.UpdateMemoryBlocks();

            PreviousOutput.Count = Neurons;
            Random.Count = Neurons;
            Bias.Count = Neurons;
            PreviousBiasDelta.Count = Neurons;
            Energy.Count = 1;

            if (Bias.Count % 2 != 0)
                Bias.Count++; // make an even number of biasses for the cuda random initialisation
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
        }


        #region Kernels

        private MyCudaKernel m_RBMInputForwardKernel;
        private MyCudaKernel m_RBMInputForwardAndStoreKernel;
        private MyCudaKernel m_RBMUpdateBiasesKernel;
        private MyCudaKernel m_RBMRandomActivationKernel;




        internal void RBMInputForward(bool partOfFeedForward = false)
        {

            // only copy to output
            if (partOfFeedForward && !ApplyBiases)
            {
                MyLog.DEBUG.WriteLine("RBM forward input without biases to " + Name);
                Input.CopyToMemoryBlock(Output, 0, 0, Neurons);
            }
            // apply biases
            else
            {
                MyLog.DEBUG.WriteLine("RBM forward input and apply biases to " + Name);
                // first calculate this layer's delta...
                m_RBMInputForwardKernel.SetupExecution(Neurons);
                m_RBMInputForwardKernel.Run(
                                    Input,
                                    Output,
                                    Bias,
                                    Convert.ToInt32(ApplyBiases),
                                    Neurons
                                    );

            }
        }

        internal void RBMInputForwardAndStore()
        {
            MyLog.DEBUG.WriteLine("RBM forward input and store to " + Name);

            // first calculate this layer's delta...
            m_RBMInputForwardAndStoreKernel.SetupExecution(Neurons);
            m_RBMInputForwardAndStoreKernel.Run(
                                Input,
                                Output,
                                Bias,
                                PreviousOutput,
                                Convert.ToInt32(ApplyBiases),
                                Neurons
                                );
        }


        internal void RBMUpdateBiases(float LearningRate, float Momentum, float WeightDecay)
        {
            MyLog.DEBUG.WriteLine("RBM bias update of " + Name);

            m_RBMUpdateBiasesKernel.SetupExecution(Neurons);
            m_RBMUpdateBiasesKernel.Run(
                                Bias,
                                PreviousOutput,
                                Output,
                                PreviousBiasDelta,
                                Energy,
                                LearningRate,
                                Momentum,
                                WeightDecay,
                                Neurons,
                                Convert.ToInt32(StoreEnergy)
                                );
        }


        /// <summary>
        /// Randomly activates the output neurons.
        /// </summary>
        internal void RBMRandomActivation()
        {
            MyLog.DEBUG.WriteLine("RBM random activation of " + Name);

            RBMGenerateRandom();

            m_RBMRandomActivationKernel.SetupExecution(Neurons);
            m_RBMRandomActivationKernel.Run(
                                Output,
                                Random,
                                Neurons
                                );
        }


        private void RBMGenerateRandom()
        {
            MyKernelFactory.Instance.GetRandDevice(this).GenerateUniform32(Random.GetDevicePtr(GPU), Random.Count);
        }
        #endregion

        public override string Description
        {
            get
            {
                return "RBM input layer";
            }
        }

        public void CreateTasks()
        {
            ForwardTask = new MyRBMInputForwardTask();
            DeltaBackTask = new MyRBMInputBackwardTask();
        }

        internal void SetOutput(float[] f)
        {
            Array.Copy(f, Input.Host, Neurons);
            Array.Copy(f, Output.Host, Neurons);
            Input.SafeCopyToDevice();
            Output.SafeCopyToDevice();
        }
    }
}
