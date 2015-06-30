using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Tasks;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace  XmlFeedForwardNet.Networks
{
    public enum MyAutoencoderMode
    {
        TRAINING,
        FORWARD_PASS,
        FEATURE_ENCODING,
        FEATURE_DECODING
    }
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Autoencoder based on a feedforward network</summary>
    /// <description>
    /// This is a special case of feed forward network. Its purpose is to learn how to reproduce the input.
    /// It can be also used for feature extraction (decoding and encoding)</description>
    public class MyAutoencoderNode : MyXMLNetNode
    {
        /****************************
        *           INPUTS
        ****************************/

        [MyInputBlock(1)]
        virtual public MyMemoryBlock<float> FeatureInput
        {
            get { return GetInput(1); }
        }

        /****************************
        *          OUTPUTS
        ****************************/

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> FeatureOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        /****************************
        *        PROPERTIES
        ****************************/

        public MyAbstractFLayer FeatureLayer { get; private set; }

        private int m_columnHint = 0;
        private MyMemoryBlock<float> lastFeatureInput = null;


        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("\tOutput"), Description("ColumnHint of output data")]
        public int ColumnHint
        {
            get
            {
                return m_columnHint;
            }

            set
            {
                if (value > 0 || LastLayer == null)
                    m_columnHint = value;
                else
                    if (XMLColumnHint > 0)
                        m_columnHint = XMLColumnHint;
                    else
                        m_columnHint = LastLayer.Output.Width;
                Output.ColumnHint = m_columnHint;
            }
        }

        /***********************
        *        TASKS
        * ********************/

        public MyInitTask InitTask { get; protected set; }
        public MyAutoencoderTask AutoencoderTask { get; protected set; }

        /****************************
        *          METHODS
        ****************************/

        public override string Description { get { return "Autoencoder"; } }

        public CUdeviceptr CurrentSampleFeatureInputPtr
        {
            get
            {
                if (FeatureInput != null && FeatureLayer != null)
                    return FeatureInput.GetDevicePtr(this, (int)(FeatureLayer.Output.Count * m_currentSamplePosition));
                else
                    return new CUdeviceptr(0);
            }
        }

        public CUdeviceptr CurrentSampleFeatureOutputPtr
        {
            get
            {
                if (FeatureOutput != null && FeatureLayer != null)
                    return FeatureOutput.GetDevicePtr(this, (int)(FeatureLayer.Output.Count * m_currentSamplePosition));
                else
                    return new CUdeviceptr(0);
            }
        }

        protected override void Dimension()
        {
            base.Dimension();

            // FeatureLayer
            if (FeatureLayerPosition >= 0 && FeatureLayerPosition < Layers.Count)
            {
                FeatureLayer = Layers[FeatureLayerPosition];
                FeatureOutput.Count = FeatureLayer.Output.Count * ForwardSamplesPerStep;
            }

            if (ColumnHint == 0 && LastLayer != null)
                ColumnHint = LastLayer.Output.Width;
        }

        public void ForwardPropagation()
        {
            switch (AutoencoderTask.NetworkMode)
            {
                case MyAutoencoderMode.TRAINING:
                case MyAutoencoderMode.FORWARD_PASS:

                    InputLayer.Forward();
                    foreach (MyAbstractFBLayer layer in Layers)
                        layer.Forward();

                    // Copy the end layer output to the node output
                    m_copyKernel.SetupExecution(LastLayer.Output.Count);
                    m_copyKernel.Run(LastLayer.Output.Ptr, 0, CurrentSampleOutputPtr, 0, LastLayer.Output.Count);
                    SamplesProcessed++;
                    break;

                case MyAutoencoderMode.FEATURE_ENCODING:

                    InputLayer.Forward();
                    for (int i = 0; i <= FeatureLayerPosition; i++)
                        Layers[i].Forward();

                    // Copy the featureLayer to feature output
                    m_copyKernel.SetupExecution(FeatureLayer.Output.Count);
                    m_copyKernel.Run(FeatureLayer.Output.Ptr, 0, CurrentSampleFeatureOutputPtr, 0, FeatureLayer.Output.Count);
                    SamplesProcessed++;
                    break;

                case MyAutoencoderMode.FEATURE_DECODING:

                    // Copy the feature input to the feature layer
                    m_copyKernel.SetupExecution(FeatureLayer.Output.Count);
                    m_copyKernel.Run(CurrentSampleFeatureInputPtr, 0, FeatureLayer.Output.Ptr, 0, FeatureLayer.Output.Count);

                    for (int i = FeatureLayerPosition + 1; i < Layers.Count; i++)
                        Layers[i].Forward();

                    // Copy the end layer output to the node output
                    m_copyKernel.SetupExecution(LastLayer.Output.Count);
                    m_copyKernel.Run(LastLayer.Output.Ptr, 0, CurrentSampleOutputPtr, 0, LastLayer.Output.Count);
                    SamplesProcessed++;
                    break;
            }
        }

        protected override void updateInput()
        {
            base.updateInput();
            if (FeatureInput != lastFeatureInput)
            {
                lastFeatureInput = FeatureInput;
                ParamsChanged = true;
            }
        }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            if (ColumnHint > 0)
                Output.ColumnHint = ColumnHint;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (!ParamsChanged)
            {
                validator.AssertError(Layers.Count > 0, this, "The network has no layers.");
                if (AutoencoderTask.NetworkMode != MyAutoencoderMode.FEATURE_DECODING)
                {
                    validator.AssertError(DataInput != null, this, "No input available.");
                    if (DataInput != null)
                    {
                        validator.AssertError(DataInput.Count > 0, this, "Input connected but empty.");
                        validator.AssertWarning(DataInput.ColumnHint > 1, this, "The Data input columnHint is 1.");
                        uint total = InputWidth * InputHeight * InputsCount * ForwardSamplesPerStep;
                        validator.AssertError(DataInput.Count == total, this, "DataInput Count is " + DataInput.Count + ". Expected " + InputLayer.Output.ToString() + " = " + total + ".");
                    }
                }

                if (AutoencoderTask.NetworkMode == MyAutoencoderMode.TRAINING)
                {
                    validator.AssertError(LastLayer != null, this, "Last layer is null.");
                }

                if (AutoencoderTask.NetworkMode == MyAutoencoderMode.FEATURE_ENCODING || AutoencoderTask.NetworkMode == MyAutoencoderMode.FEATURE_DECODING)
                {
                    validator.AssertError(FeatureLayer != null, this, "In FEATURE_ENCODING or FEATURE_DECODING mode, a featureLayer must be present in the network architecture");
                }

                if (AutoencoderTask.NetworkMode == MyAutoencoderMode.FEATURE_ENCODING)
                {
                    if (FeatureOutput != null && FeatureLayer != null)
                        validator.AssertError(FeatureOutput.Count == FeatureLayer.Output.Count * ForwardSamplesPerStep, this, "In FEATURE_DECODING mode, the Feature output must have the same size (currently " + FeatureOutput.Count + ") as the featureLayer output (" + FeatureLayer.Output.Count + " x " + ForwardSamplesPerStep + ")");
                }
                if (AutoencoderTask.NetworkMode == MyAutoencoderMode.FEATURE_DECODING)
                {
                    validator.AssertError(FeatureInput != null, this, "In FEATURE_DECODING mode, the Feature input must be connected");
                    if (FeatureInput != null && FeatureLayer != null)
                        validator.AssertError(FeatureInput.Count == FeatureLayer.Output.Count * ForwardSamplesPerStep, this, "In FEATURE_DECODING mode, the Feature input must have the same size (currently " + FeatureInput.Count + ") as the featureLayer output (" + FeatureLayer.Output.Count + " x " + ForwardSamplesPerStep + ")");
                }
            }
        }
    }
}
