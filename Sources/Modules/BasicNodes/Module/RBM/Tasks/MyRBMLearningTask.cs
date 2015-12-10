using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.RBM
{

    /// <summary>
    ///     Learning of the whole RBM group.
    /// <p>
    /// Specify the layer to be learned with CurrentLayerIndex parameter.
    /// Layers are indexed from 0 (zero).
    /// </p>
    /// <p>
    /// Current layer 0 means we are learning weights between the 0th and 1st layers (i. e. the first two layers).
    /// Typically, you want to learn the RBM layer-wise starting from 0.
    /// Start with layer index of 0, after first weights (between 0 and 1 are learned), increase it to 1, etc., until you reach (last but one)th layer.
    /// </p>
    /// <p>
    /// Use RBMFilterObserver (upper right by default) to see weights.
    /// </p>
    /// </summary>
    [Description("RBM Learning"), MyTaskInfo(OneShot = false)]
    public class MyRBMLearningTask : MyAbstractBackpropTask
    {
        #region Layer indexing
        [YAXSerializableField(DefaultValue = 0)]
        private int m_layerIndex = 0;
        [MyBrowsable, Category("\tIndexing"), Description("Index of current layer.")]
        public int CurrentLayerIndex
        {
            get { return m_layerIndex; }
            set
            {
                if (value < 0)
                    return;
                step = 0;
                m_layerIndex = value;
            }
        }

        #endregion

        #region Learning parameters

        [YAXSerializableField(DefaultValue = 1)]
        private int m_cdk;
        [MyBrowsable, Category("\tLearning"), Description("k in CD-k algorithm.")]
        public int CD_k
        {
            get { return m_cdk; }
            set
            {
                if (value < 0)
                    return;
                m_cdk = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0.01f)]
        private float m_learningRate;
        [MyBrowsable, Category("\tLearning"), Description("Factor of weight changes.")]
        public float LearningRate
        {
            get { return m_learningRate; }
            set
            {
                if (value < 0)
                    return;
                m_learningRate = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0.9f)]
        private float m_momentum;
        [MyBrowsable, Category("\tLearning"), Description("Momentum parameter.")]
        public float Momentum
        {
            get { return m_momentum; }
            set
            {
                if (value < 0)
                    return;
                m_momentum = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0f)]
        private float m_weightDecay;
        [MyBrowsable, Category("\tLearning"), Description("Weight decay.")]
        public float WeightDecay
        {
            get { return m_weightDecay; }
            set
            {
                if (value < 0)
                    return;
                m_weightDecay = value;
            }
        }
        #endregion

        #region Activation rules


        [YAXSerializableField(DefaultValue = 0.5f)]
        private float m_sigmoidSteepness = 0.5f;
        [MyBrowsable, Category("\tActivation"), Description("Steepnes of the sigmoid function.")]
        public float SigmoidSteepness
        {
            get { return m_sigmoidSteepness; }
            set
            {
                if (value < 0)
                    return;
                m_sigmoidSteepness = value;
            }
        }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tActivation"), Description("Is activation of the visible neurons random or determined by possibilities?")]
        public bool RandomVisible
        {
            get;
            set;
        }

        [YAXSerializableField(DefaultValue = true)]
        [MyBrowsable, Category("\tActivation"), Description("Is activation of the hidden neurons random?")]
        public bool RandomHidden
        {
            get;
            set;
        }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tActivation"), Description("Is activation of the previous layers (with index < CurrentVisibleLayer) random?")]
        public bool RandomPrevious
        {
            get;
            set;
        }
        #endregion


        //parameterless constructor
        public MyRBMLearningTask() { }

        //put additional kernels here

        private int step;
        MyRBMLayer hidden;
        List<MyAbstractLayer> layers;
        MyRBMInputLayer InputLayer;


        //Kernel initialization
        public override void Init(int nGPU)
        {
            layers = new List<MyAbstractLayer>();

            if (Owner.SortedChildren == null)
                Owner.InitGroup.Execute(); // TODO - horrible hack

            foreach (var i in Owner.SortedChildren)
            {
                if (i is MyRBMInputLayer || i is MyRBMLayer)
                {
                    layers.Add((MyAbstractLayer)i);
                    if (i is MyRBMInputLayer)
                        ((MyRBMInputLayer)i).Init(nGPU);
                    else
                        ((MyRBMLayer)i).Init(nGPU);
                }
            }

            InputLayer = (MyRBMInputLayer)layers[0];

            step = 0;
        }

        //Task execution
        public override void Execute()
        {

            // sampling between input and hidden layer
            if (CurrentLayerIndex <= 0)
            {
                InputLayer.RBMInputForwardAndStore();

                MyRBMInputLayer visible = InputLayer;
                hidden = (MyRBMLayer)layers[CurrentLayerIndex + 1];

                if (hidden.Dropout > 0)
                    hidden.CreateDropoutMask();

                // forward to visible and hidden layers and store for sampling biases
                hidden.RBMForwardAndStore(SigmoidSteepness);

                // sample positive weight data
                hidden.RBMSamplePositive();

                if (RandomHidden)
                    hidden.RBMRandomActivation();

                // do k-step Contrastive Divergence (go back and forth between visible and hidden)
                for (int i = 0; i < CD_k - 1; i++)
                {
                    // back
                    hidden.RBMBackward(visible.Bias, SigmoidSteepness);


                    if (RandomVisible)
                        visible.RBMRandomActivation();

                    // and forth
                    hidden.RBMForward(SigmoidSteepness, true);

                    // randomly activate the just updated hidden neurons if needed
                    if (RandomHidden)
                        hidden.RBMRandomActivation();
                }

                // in the last (= k-th) step of CD-k, we use probabilistic modeling -> we sample both with probability, not random activation
                hidden.RBMBackward(visible.Bias, SigmoidSteepness);
                hidden.RBMForward(SigmoidSteepness, true);

                if (hidden.StoreEnergy)
                    hidden.Energy.Fill(0);

                // update biases of both layers based on the sampled data stored by RBMForwardAndStore()
                visible.RBMUpdateBiases(LearningRate, Momentum, WeightDecay);
                hidden.RBMUpdateBiases(LearningRate, Momentum, WeightDecay);

                // sample negative weight data AND adapt weights at the same time
                hidden.RBMUpdateWeights(LearningRate, Momentum, WeightDecay);

            }

            // sampling between hidden layers

            else if (layers.Count > 2) // if number of  hidden layers is greater than 1)
            {
                InputLayer.RBMInputForward();

                // hidden layers, up to the one that will be visible (excluding)
                for (int i = 1; i < CurrentLayerIndex; i++)
                {
                    ((MyRBMLayer)layers[i]).RBMForward(SigmoidSteepness, false);
                    if (RandomPrevious)
                        ((MyRBMLayer)layers[i]).RBMRandomActivation();
                }

                MyRBMLayer visible = (MyRBMLayer)layers[CurrentLayerIndex];
                hidden = (MyRBMLayer)layers[CurrentLayerIndex + 1];

                if (hidden.Dropout > 0)
                    hidden.CreateDropoutMask();

                // forward to visible and hidden layers and store for sampling biases
                visible.RBMForwardAndStore(SigmoidSteepness);
                hidden.RBMForwardAndStore(SigmoidSteepness);

                // sample positive weight data
                hidden.RBMSamplePositive();

                if (RandomHidden)
                    hidden.RBMRandomActivation();

                // do k-step Contrastive Divergence (go back and forth between visible and hidden)
                for (int i = 0; i < CD_k - 1; i++)
                {
                    // back
                    hidden.RBMBackward(visible.Bias, SigmoidSteepness);


                    if (RandomVisible)
                        visible.RBMRandomActivation();

                    // and forth
                    hidden.RBMForward(SigmoidSteepness, true);

                    // randomly activate the just updated hidden neurons if needed
                    if (RandomHidden)
                        hidden.RBMRandomActivation();
                }

                // in last (= k-th) step of CD-k, we use probabilistic modeling -> we sample both with probability, not random activation
                hidden.RBMBackward(visible.Bias, SigmoidSteepness);
                hidden.RBMForward(SigmoidSteepness, true);

                if (hidden.StoreEnergy)
                    hidden.Energy.Fill(0);

                // update biases of both layers based on the sampled data stored by RBMForwardAndStore()
                visible.RBMUpdateBiases(LearningRate, Momentum, WeightDecay);
                hidden.RBMUpdateBiases(LearningRate, Momentum, WeightDecay);

                // sample negative weight data AND adapt weights at the same time
                hidden.RBMUpdateWeights(LearningRate, Momentum, WeightDecay);

            }
            else
            {
                MyLog.ERROR.WriteLine("Wrong CurrentLayerIndex parameter. There are " + layers.Count + " total layers, can't sample from " + CurrentLayerIndex);
            }


            MyLog.DEBUG.WriteLine("RBM initialization between layers [" + CurrentLayerIndex + ";" + (CurrentLayerIndex + 1) + "], step " + step + ".");
            ++step;
        }

    }

}