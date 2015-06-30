using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Signals;
using GoodAI.Core.Utils;
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
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Feedforward network with different inputs for classification and learning</summary>
    /// <description>
    /// This is a special case of a feed forward network. It has two inputs.
    /// One for classification and one for learning phase. Both these phases
    /// can be turned on and off by external signal</description>
    public class MyDynamicFFNode : MyXMLNetNode
    {
        /****************************
        *           INPUTS
        ****************************/

        [MyInputBlock(1)]
        public MyMemoryBlock<float> TrainingLabel
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> TrainingData
        {
            get { return GetInput(2); }
        }

        /****************************
        *        PROPERTIES
        ****************************/

        private int m_columnHint = 0;

        private MyMemoryBlock<float> lastTrainingLabel = null;
        private MyMemoryBlock<float> lastTrainingData = null;

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

        public MyForwardSignal ForwardSignal { get; private set; }
        public class MyForwardSignal : MySignal { }

        public MyTrainingSignal TrainingSignal { get; private set; }
        public class MyTrainingSignal : MySignal { }

        /***********************
        *        TASKS
        * ********************/

        public MyInitTask InitTask { get; protected set; }
        public MyDynamicFFTask DynamicFFTask { get; protected set; }

        /****************************
        *          METHODS
        ****************************/

        public override string Description { get { return "Dynamic"; } }

        protected override void Dimension()
        {
            base.Dimension();

            if (ColumnHint == 0 && LastLayer != null)
                ColumnHint = LastLayer.Output.Width;
        }

        public void ForwardPropagation()
        {
            InputLayer.Forward();
            for (int i = 0; i < Layers.Count; i++)

                Layers[i].Forward();
        }

        public void CopyResult()
        {
            // Copy the end layer output to the node output
            m_copyKernel.SetupExecution(LastLayer.Output.Count);
            m_copyKernel.Run(LastLayer.Output.Ptr, 0, CurrentSampleOutputPtr, 0, LastLayer.Output.Count);
            SamplesProcessed++;
        }

        protected override void updateInput()
        {
            base.updateInput();
            if (TrainingLabel != lastTrainingLabel)
            {
                lastTrainingLabel = TrainingLabel;
                ParamsChanged = true;
            }
            if (TrainingData != lastTrainingData)
            {
                lastTrainingData = TrainingData;
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

                validator.AssertError(TrainingData != null, this, "No TrainingData available.");
                if (TrainingData != null)
                {
                    validator.AssertError(TrainingData.Count > 0, this, "TrainingData connected but empty.");
                    validator.AssertWarning(TrainingData.ColumnHint > 1, this, "TrainingData columnHint is 1.");
                    uint total = InputWidth * InputHeight * InputsCount * TrainingSamplesPerStep;
                    validator.AssertError(TrainingData.Count == total, this, "TrainingData Count is " + TrainingData.Count + ". Expected " + InputLayer.Output.ToString() + " = " + total + ".");
                }

                validator.AssertError(TrainingLabel != null, this, "No TrainingLabel available.");
                validator.AssertError(LastLayer != null, this, "Last layer is null.");
                if (TrainingLabel != null && LastLayer != null)
                    validator.AssertError(TrainingLabel.Count == LastLayer.Output.Count * TrainingSamplesPerStep, this, "Current label dimension is " + TrainingLabel.Count + ". Expected " + TrainingSamplesPerStep + "x" + LastLayer.Output.Count + ".");
            }
        }
    }
}
