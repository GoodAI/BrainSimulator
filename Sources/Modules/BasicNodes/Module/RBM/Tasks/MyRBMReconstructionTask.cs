using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.RBM
{
    public enum ReconstructionSource
    {
        INPUT,
        HIDDEN
    }

    /// <summary>
    /// <p>
    /// Reconstruction task for RBM.
    /// </p>
    /// <p>
    /// Layers are indexed from 0 (zero).
    /// </p>
    /// 
    /// Can reconstruct from
    /// <ul>
    ///     <li> input up to the layer specified by CurrentLayerIndex.</li>
    ///     <li> layer specified by CurrentLayerIndex back towards the first (input) (0th) layer.</li>
    /// </ul>
    /// </summary>
    [Description("RBM Reconstruction"), MyTaskInfo(OneShot = false)]
    public class MyRBMReconstructionTask : MyAbstractBackpropTask
    {

        #region Layer indexing
        [YAXSerializableField(DefaultValue = 0)]
        private int m_layerIndex = 0;
        [MyBrowsable, Category("\tIndexing"), Description("Index of current VISIBLE layer.")]
        public int CurrentLayerIndex
        {
            get { return m_layerIndex; }
            set
            {
                if (value < 0)
                    return;
                if (layers != null && value >= layers.Count)
                    m_layerIndex = 0;
                step = 0;
                m_layerIndex = value;
            }
        }

        #endregion

        #region Reconstruction parameters

        [YAXSerializableField(DefaultValue = 1)]
        private int cdk = 1;
        [MyBrowsable, DisplayName("Iterations"), Category("\tReconstruction"), Description("Corresponds to k in CD-k algorithm.")]
        public int CD_k
        {
            get { return cdk; }
            set
            {
                if (value < 0)
                    return;
                cdk = value;
            }
        }

        [YAXSerializableField(DefaultValue = ReconstructionSource.INPUT)]
        [MyBrowsable, DisplayName("Reconstruction source"), Category("\tReconstruction"), Description("INPUT to reconstruct from the RBM group input.")]
        public ReconstructionSource ReconstructionSource { get; set; }

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
        [MyBrowsable, Category("\tActivation"), Description("Is activation of the visible neurons random or equal to probability?")]
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


        private int step;
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

            if (layers.Count <= CurrentLayerIndex)
            {
                MyLog.ERROR.WriteLine("Invalid CurrentLayerIndex " + CurrentLayerIndex +
                                      ". Must be smaller than number of layers which is " + layers.Count);
                return;
            }
            if (ReconstructionSource == ReconstructionSource.INPUT)
            {

                InputLayer.RBMInputForward();

                for (int i = 0; i < CD_k; i++)
                {
                    RBMForwardPass(CurrentLayerIndex, i >= (CD_k - 1));

                    RBMBackwardPass(CurrentLayerIndex, i >= (CD_k - 1));
                }
            }
            else
            {
                // we want to reconstruct from hidden, set layer index to first hidden if it was set to visible
                if (CurrentLayerIndex == 0)
                    CurrentLayerIndex = 1;

                MyRBMLayer layer = ((MyRBMLayer)layers[CurrentLayerIndex]);
                if (layer.Target != null && layer.Target.Count > 0)
                    layer.Output.CopyFromMemoryBlock(layer.Target, 0, 0, layer.Neurons);

                RBMBackwardPass(CurrentLayerIndex, false);

                for (int i = 0; i < CD_k; i++)
                {
                    RBMForwardPass(CurrentLayerIndex, i >= (CD_k - 1));
                    RBMBackwardPass(CurrentLayerIndex, i >= (CD_k - 1));
                }
            }

            MyLog.DEBUG.WriteLine("RBM reconstruction between layers [" + CurrentLayerIndex + ";" + (CurrentLayerIndex + 1) + "], step " + step + ".");
            ++step;
        }

        private void RBMForwardPass(int to, bool last)
        {
            for (int i = 1; i < to; i++)
            {
                ((MyRBMLayer)layers[i]).RBMForward(SigmoidSteepness, false);
                if (RandomPrevious)
                    ((MyRBMLayer)layers[i]).RBMRandomActivation();

            }
            if (to < 1)
                to = 1;
            ((MyRBMLayer)layers[to]).RBMForward(SigmoidSteepness, false);
            if (/*!last && */RandomHidden)
                ((MyRBMLayer)layers[to]).RBMRandomActivation();

        }

        private void RBMBackwardPass(int to, bool last)
        {
            for (int i = to; i > 1; --i)
            {
                ((MyRBMLayer)layers[i]).RBMBackward(((MyRBMLayer)layers[i - 1]).Bias, SigmoidSteepness);
                if (RandomPrevious)
                    ((MyRBMLayer)layers[i-1]).RBMRandomActivation();
            }

            // last (or first) pair of layers (first hidden layer to input layer)
            ((MyRBMLayer)layers[1]).RBMBackward(((MyRBMInputLayer)layers[0]).Bias, SigmoidSteepness);
            if (/*!last && */RandomVisible)
                ((MyRBMInputLayer)layers[0]).RBMRandomActivation();
        }
    }

}
