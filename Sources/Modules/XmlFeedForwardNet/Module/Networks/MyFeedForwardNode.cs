using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System.ComponentModel;
using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Tasks;
using XmlFeedForwardNet.Tasks.RBM;
using YAXLib;

namespace  XmlFeedForwardNet.Networks
{
    public enum MyFeedForwardMode
    {
        TRAINING,
        FORWARD_PASS
    }

    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Classic feedforward network with xml-defined topology</summary>
    /// <description>
    /// This is the classic feed forward network. You can define its topology
    /// in an external xml file.</description>
    public class MyFeedForwardNode : MyXMLNetNode
    {
        /****************************
        *           INPUTS
        ****************************/

        [MyInputBlock(1)]
        public MyMemoryBlock<float> LabelInput
        {
            get { return GetInput(1); }
        }

        /****************************
        *        PROPERTIES
        ****************************/

        private int m_columnHint = 0;

        private MyMemoryBlock<float> lastLabelInput = null;

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

        /**********************
        *        TASKS
        * ********************/

        public MyInitTask InitTask { get; protected set; }
        public MyFBPropTask FBPropTask { get; protected set; }
        public MyRBMTask RBMTask { get; protected set; }

        /****************************
        *          METHODS
        ****************************/

        public override string Description { get { return "FeedForward"; } }

        protected override void Dimension()
        {
            base.Dimension();

            if (ColumnHint == 0 && LastLayer != null)
                ColumnHint = LastLayer.Output.Width;
        }

        public void ForwardPropagation()
        {
            InputLayer.Forward();
            foreach (MyAbstractFBLayer layer in Layers)
                layer.Forward();

            // Copy the end layer output to the node output
            m_copyKernel.SetupExecution(LastLayer.Output.Count);
            m_copyKernel.Run(LastLayer.Output.Ptr, 0, CurrentSampleOutputPtr, 0, LastLayer.Output.Count);
            SamplesProcessed++;
        }

        protected override void updateInput()
        {
            base.updateInput();
            if (LabelInput != lastLabelInput)
            {
                lastLabelInput = LabelInput;
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

                validator.AssertError(DataInput != null, this, "No input available.");
                if (DataInput != null)
                {
                    validator.AssertError(DataInput.Count > 0, this, "Input connected but empty.");
                    validator.AssertWarning(DataInput.ColumnHint > 1, this, "The Data input columnHint is 1.");
                    uint total = InputWidth * InputHeight * InputsCount * ForwardSamplesPerStep;
                    validator.AssertError(DataInput.Count == total, this, "DataInput Count is " + DataInput.Count + ". Expected " + InputLayer.Output.ToString() + " = " + total + ".");
                }

                if (FBPropTask.NetworkMode == MyFeedForwardMode.TRAINING)
                {
                    validator.AssertError(LabelInput != null, this, "No label available.");
                    validator.AssertError(LastLayer != null, this, "Last layer is null.");
                    if (LabelInput != null && LastLayer != null)
                        validator.AssertError(LabelInput.Count == LastLayer.Output.Count * ForwardSamplesPerStep, this, "Current label dimension is " + LabelInput.Count + ". Expected " + ForwardSamplesPerStep + "x" + LastLayer.Output.Count + ".");
                }
            }
        }
    }
}