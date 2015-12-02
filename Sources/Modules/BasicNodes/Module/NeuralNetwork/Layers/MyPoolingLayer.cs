using CustomModels.NeuralNetwork.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Layers
{

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>Pooling layer.</summary>
    /// <description>Layer that performs max pooling. It takes max value of each input window and presents it to the output.</description>
    public class MyPoolingLayer : MyAbstractLayer, IMyCustomTaskFactory
    {

        public MyMemoryBlock<float> ActivatedNeurons { get; protected set; }

        public override ConnectionType Connection
        {
            get { return ConnectionType.ONE_TO_ONE; }
        }

        [YAXSerializableField(DefaultValue = 128)]
        [MyBrowsable, Category("\tLayer"), ReadOnly(true)]
        public override int Neurons { get; set; }


        #region Input
        [MyBrowsable, Category("Input image dimensions"), DisplayName("\t\tWidth"), ReadOnly(true)]
        public int InputWidth
        {
            get { return MyConvolutionLayer.GetInputDimension(Input, 0); }

        }


        [MyBrowsable, Category("Input image dimensions"), DisplayName("\tHeight"), ReadOnly(true)]
        public int InputHeight
        {
            get { return MyConvolutionLayer.GetInputDimension(Input, 1); }

        }


        [MyBrowsable, Category("Input image dimensions"), DisplayName("Depth"), ReadOnly(true)]
        public int InputDepth
        {
            get { return MyConvolutionLayer.GetInputDimension(Input, 2); }

        }
        #endregion


        #region Output
        [MyBrowsable, Category("Output image dimensions"), DisplayName("\t\tWidth"), ReadOnly(true)]
        public int OutputWidth
        {
            get { return MyConvolutionLayer.SetOutputDimension(InputWidth, FilterWidth, HorizontalStride); }
        }

        [MyBrowsable, Category("Output image dimensions"), DisplayName("\tHeight"), ReadOnly(true)]
        public int OutputHeight
        {
            get { return MyConvolutionLayer.SetOutputDimension(InputHeight, FilterHeight, VerticalStride); }
        }

        [MyBrowsable, Category("Output image dimensions"), DisplayName("Depth"), ReadOnly(true)]
        public int OutputDepth
        {
            get { return InputDepth; }
        }
        #endregion


        #region Filter
        [YAXSerializableField(DefaultValue = 2)]
        private int m_width = 2;
        [MyBrowsable, Category("Filter"), DisplayName("\t\t\tWidth")]
        public int FilterWidth
        {
            get { return m_width; }
            set
            {
                if (value < 1)
                    return;
                m_width = value;
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        private int m_height = 2;
        [MyBrowsable, Category("Filter"), DisplayName("\t\tHeight")]
        public int FilterHeight
        {
            get { return m_height; }
            set
            {
                if (value < 1)
                    return;
                m_height = value;
            }
        }


        [YAXSerializableField(DefaultValue = 2)]
        private int m_horizontalStride = 2;
        [MyBrowsable, Category("Filter"), DisplayName("Horizontal stride")]
        public int HorizontalStride
        {
            get { return m_horizontalStride; }
            set
            {
                if (value < 1)
                    return;
                m_horizontalStride = value;
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        private int m_verticalStride = 2;
        [MyBrowsable, Category("Filter"), DisplayName("Vertical stride")]
        public int VerticalStride
        {
            get { return m_verticalStride; }
            set
            {
                if (value < 1)
                    return;
                m_verticalStride = value;
            }
        }




        #endregion

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            Neurons = OutputWidth*OutputHeight*OutputDepth;
            base.UpdateMemoryBlocks();
            if (Neurons > 0)
            {
                Output.Dims = new TensorDimensions(OutputWidth, OutputHeight, OutputDepth);
                ActivatedNeurons.Count = Neurons;
                Delta.Dims = Output.Dims;
                ActivatedNeurons.Dims = Output.Dims;
            }
        }

        //Validation rules
        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (PreviousTopologicalLayer != null)
                validator.AssertError(InputWidth * InputHeight * InputDepth == PreviousTopologicalLayer.Neurons, this, "'Input width * input height * depth' must be equal to output size of the previous layer.");

            validator.AssertError((InputWidth - FilterWidth) % HorizontalStride == 0, this, "Filter does not fit the input image horizontally when striding.");
            
            // horizontal input check:
            validator.AssertError((InputHeight - FilterHeight) % VerticalStride == 0, this, "Filter does not fit the input image vertically when striding.");

            // vertical input check:
            validator.AssertError(InputHeight > FilterHeight && InputWidth > FilterWidth, this, "Filter dimensions must be smaller than input dimensions.");


        }

        // description
        public override string Description
        {
            get
            {
                return "Pooling layer";
            }
        }

        public void CreateTasks()
        {
            ForwardTask = new MyPoolingForwardTask();
            DeltaBackTask = new MyPoolingBackwardTask();
        }
    }
}
