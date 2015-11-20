using GoodAI.Core;
using GoodAI.Core.Memory;
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
    ///     One layer of Restricted Boltzmann Machine network.
    ///     Inherited from classic neural hidden layer.
    ///     Can act as both visible and hidden layer.
    /// </summary>
    /// <description>
    /// <p>Specify the layer to be learned with CurrentLayerIndex parameter.</p>
    /// <p>Layers are indexed from 0 (zero).</p>
    /// <br/>
    /// 
    /// <p>Current layer 0 means we are learning weights between the 0th and 1st layers (i. e. the first two layers).</p>
    /// <p>Typically, you want to learn the RBM layer-wise starting from 0.</p>
    /// <p>Start with layer index of 0, after first weights (between 0 and 1) are learned, increase it to 1, etc., until you reach (last but one)th layer.</p>
    /// <br/>
    /// <p>Use RBMFilterObserver (upper right by default) to see weights.</p>
    /// </description>
    public class MyRBMLayer : MyOutputLayer
    {

        public MyRBMInitLayerTask RBMInitLayerTask { get; protected set; }
        public MyRBMRandomWeightsTask RBMRandomWeightsTask { get; protected set; }


        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tLayer")]
        [ReadOnly(false)]
        public override int Neurons { get; set; }


        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tLayer")]
        public bool StoreEnergy { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        private float m_dropout = 0;
        [MyBrowsable, Category("\tLayer"), DisplayName("RBM Dropout")]
        public float Dropout
        {
            get { return m_dropout; }
            set
            {
                if (value < 0 || value > 1)
                    return;
                m_dropout = value;
            }
        }

        // Memory blocks

        public MyMemoryBlock<float> PreviousOutput { get; protected set; }
        public MyMemoryBlock<float> RBMWeightPositive { get; protected set; }
        public MyMemoryBlock<float> Filter { get; protected set; }
        public MyMemoryBlock<float> Random { get; protected set; }
        public MyMemoryBlock<float> Energy { get; protected set; }

        public void Init(int nGPU)
        {
            m_RBMForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMForwardKernel");
            m_RBMForwardAndStoreKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMForwardAndStoreKernel");
            m_RBMBackwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMBackwardKernel");
            m_RBMSamplePositiveKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMSamplePositiveKernel");
            m_RBMUpdateWeightsKernel  = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMUpdateWeightsKernel");
            m_RBMCopyFilterKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMCopyFilterKernel");
            m_RBMUpdateBiasesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMUpdateBiasesKernel");
            m_RBMRandomActivationKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMRandomActivationKernel");
            m_RBMDropoutMaskKernel = MyKernelFactory.Instance.Kernel(nGPU, @"RBM\RBMKernels", "RBMDropoutMaskKernel");

        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            if (Neurons > 0)
            {
                Random.Count = Neurons;
                PreviousOutput.Count = Neurons;
                Energy.Count = 1;
                // allocate memory scaling with input
                if (Input != null)
                {
                    Filter.Count = Input.Count;
                    Filter.ColumnHint = (int)Math.Sqrt((double)Input.Count);
                    
                    RBMWeightPositive.Count = Neurons * Input.Count;
                    Weights.Count += Weights.Count%2;
                }
                if (Target != null)
                {
                    Target.Count = Neurons;
                }
            }
        }

        public override void Validate(MyValidator validator)
        {
            //base.Validate(validator);
            validator.AssertError(Neurons > 0, this, "Number of neurons should be > 0");
            validator.AssertError(Input != null, this, "Neural network node \"" + this.Name + "\" has no input.");
            validator.AssertWarning(Connection != ConnectionType.NOT_SET, this, "ConnectionType not set for " + this);
        }

        #region Kernels

        private MyCudaKernel m_RBMForwardKernel;
        private MyCudaKernel m_RBMForwardAndStoreKernel;
        private MyCudaKernel m_RBMBackwardKernel;
        private MyCudaKernel m_RBMSamplePositiveKernel;
        private MyCudaKernel m_RBMUpdateWeightsKernel;
        private MyCudaKernel m_RBMCopyFilterKernel;
        private MyCudaKernel m_RBMUpdateBiasesKernel;
        private MyCudaKernel m_RBMRandomActivationKernel;
        private MyCudaKernel m_RBMDropoutMaskKernel;

        internal void RBMForward(float SigmoidSteepness, bool useDropoutMask)
        {
            MyLog.DEBUG.WriteLine("RBM forward to " + Name);
            m_RBMForwardKernel.SetupExecution(Neurons);
            m_RBMForwardKernel.Run(
                                Input,
                                Output,
                                Weights,
                                Bias,
                                SigmoidSteepness,
                                Input.Count,
                                Neurons,
                                Convert.ToInt32(useDropoutMask),
                                Convert.ToInt32(Dropout > 0),
                                1-Dropout,
                                DropoutMask
                                );
        }

        internal void RBMForwardAndStore(float SigmoidSteepness)
        {
            MyLog.DEBUG.WriteLine("Forwarding and storing to " + Name);
            m_RBMForwardAndStoreKernel.SetupExecution(Neurons);
            m_RBMForwardAndStoreKernel.Run(
                                Input,
                                Output,
                                Weights,
                                Bias,
                                PreviousOutput,
                                SigmoidSteepness,
                                Input.Count,
                                Neurons,
                                Convert.ToInt32(Dropout > 0),
                                DropoutMask
                                );
        }

        internal void RBMSamplePositive()
        {
            
            MyLog.DEBUG.WriteLine("Sampling positive weights of " + Name);

            m_RBMSamplePositiveKernel.SetupExecution(Weights.Count);
            m_RBMSamplePositiveKernel.Run(
                                    Input,
                                    Output,
                                    RBMWeightPositive,
                                    Neurons,
                                    Weights.Count
                                    );
        }

        internal void RBMBackward(MyMemoryBlock<float> PreviousLayerBias, float SigmoidSteepness)
        {
            MyLog.DEBUG.WriteLine("RBM backward from " + Name);

            m_RBMBackwardKernel.SetupExecution(Input.Count);
            m_RBMBackwardKernel.Run(
                                Output,
                                Input,
                                Weights,
                                PreviousLayerBias,
                                SigmoidSteepness,
                                Input.Count,
                                Neurons
                                );
        }

        internal void RBMUpdateWeights(float LearningRate, float Momentum, float WeightDecay)
        {
            MyLog.DEBUG.WriteLine("RBM updating weights of " + Name);

            m_RBMUpdateWeightsKernel.SetupExecution(Weights.Count);
            m_RBMUpdateWeightsKernel.Run(
                                Input,
                                Output,
                                Weights,
                                RBMWeightPositive,
                                PreviousWeightDelta,
                                Energy,
                                LearningRate,
                                Momentum,
                                WeightDecay,
                                Neurons,
                                Weights.Count,
                                Convert.ToInt32(StoreEnergy)
                                );
        }

        internal void RBMCopyFilter(int FilterIndex)
        {
            MyLog.DEBUG.WriteLine("RBM copying filter " + FilterIndex);

            m_RBMCopyFilterKernel.SetupExecution(Input.Count);
            m_RBMCopyFilterKernel.Run(
                                Weights,
                                Filter,
                                Input.Count,
                                FilterIndex,
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
            MyKernelFactory.Instance.GetRandDevice(this).GenerateUniform32(Random.GetDevicePtr(GPU), Neurons);
        }

        #endregion

        public override string Description
        {
            get
            {
                return "RBM Layer";
            }
        }

        internal void SetOutput(float[] f)
        {
            Array.Copy(f, Output.Host, Neurons);
            Output.SafeCopyToDevice();
        }

        public new void CreateDropoutMask()
        {
            MyLog.DEBUG.WriteLine("RBM dropout mask creation of " + Name);

            MyKernelFactory.Instance.GetRandDevice(this).GenerateUniform32(DropoutMask.GetDevicePtr(GPU), Neurons);

            m_RBMDropoutMaskKernel.SetupExecution(Neurons);
            m_RBMDropoutMaskKernel.Run(
                DropoutMask,
                Dropout,
                Neurons
                );

        }
    }
}