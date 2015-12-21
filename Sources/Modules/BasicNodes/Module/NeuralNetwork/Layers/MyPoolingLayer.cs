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

        #region Parameters

        public override ConnectionType Connection
        {
            get { return ConnectionType.ONE_TO_ONE; }
        }

        [YAXSerializableField(DefaultValue = 128)]
        [MyBrowsable, Category("\tLayer"), ReadOnly(true)]
        public override int Neurons { get; set; }

//        [YAXSerializableField(DefaultValue = ActivationFunction.MAX)]
//        [MyBrowsable, Category("Activation")]
//        public ActivationFunction ActivationFunction { get; set; }



        [YAXSerializableField(DefaultValue = false)]
        private bool m_autoInput = false;
        [MyBrowsable, Category("Input image dimensions"), DisplayName("Automatic parameters")]
        public bool AutomaticInput
        {
            get { return m_autoInput; }
            set
            {
                m_autoInput = false;
                if (value)
                {
                    if (InputConnections.Length <= 0)
                    {
                        MyLog.ERROR.WriteLine("No connected layers to layer " + this.Name + ". Please update memory blocks.");
                        return;
                    }
                    if (Input.Owner is MyAbstractLayer)
                    {
                        MyAbstractLayer prevLayer = Input.Owner as MyAbstractLayer;

                        if (prevLayer is MyConvolutionLayer)
                        {
                            InputWidth = ((MyConvolutionLayer)prevLayer).OutputWidth;
                            InputHeight = ((MyConvolutionLayer)prevLayer).OutputHeight;
                            Depth = ((MyConvolutionLayer)prevLayer).FilterCount;
                            MyLog.INFO.WriteLine("Input parameters of layer " + this.Name + " were set succesfully.");
                            return;
                        }

                        else if (prevLayer is MyPoolingLayer)
                        {
                            InputWidth = ((MyPoolingLayer)prevLayer).OutputWidth;
                            InputHeight = ((MyPoolingLayer)prevLayer).OutputHeight;
                            Depth = ((MyPoolingLayer)prevLayer).Depth;
                            MyLog.INFO.WriteLine("Input parameters of layer " + this.Name + " were set succesfully.");
                            return;
                        }

                        // assume input image
                        if ((prevLayer is MyHiddenLayer) || (Input != null && Input.Count > 0))
                        {
                            MyLog.INFO.WriteLine("Layer " + this.Name +
                                                 " cannot interpret input neuron count as image width, height and depth automatically.");
                            return;
                        }
                    }

                    MyLog.WARNING.WriteLine("Input parameters for layer " + this.Name +
                                            " could not be set automatically.");
                }
            }
        }


        [YAXSerializableField(DefaultValue = 1)]
        private int m_depth = 1;
        [MyBrowsable, Category("Input image dimensions")]
        public int Depth
        {
            get { return m_depth; }
            set
            {
                if (value < 1)
                    return;
                m_depth = value;
            }
        }


        [YAXSerializableField(DefaultValue = 2)]
        private int m_inputWidth = 2;
        [MyBrowsable, Category("Input image dimensions")]
        public int InputWidth
        {
            get { return m_inputWidth; }
            set
            {
                if (value < 1)
                    return;
                m_inputWidth = value;
                OutputWidth = MyConvolutionLayer.SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, 0);
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        private int m_inputHeight = 2;
        [MyBrowsable, Category("Input image dimensions")]
        public int InputHeight
        {
            get { return m_inputHeight; }
            set
            {
                if (value < 1)
                    return;
                m_inputHeight = value;
                OutputHeight = MyConvolutionLayer.SetOutputDimension(InputHeight, FilterHeight, VerticalStride, 0);
            }
        }


        [YAXSerializableField(DefaultValue = 0)]
        private int m_outputWidth = 0;
        [MyBrowsable, Category("Output image dimensions"), ReadOnly(true)]
        public int OutputWidth
        {
            get { return m_outputWidth; }
            set
            {
                if (value < 1)
                    return;
                m_outputWidth = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0)]
        private int m_outputHeight = 0;
        [MyBrowsable, Category("Output image dimensions"), ReadOnly(true)]
        public int OutputHeight
        {
            get { return m_outputHeight; }
            set
            {
                if (value < 1)
                    return;
                m_outputHeight = value;
            }
        }



        [YAXSerializableField(DefaultValue = 2)]
        private int m_width = 2;
        [MyBrowsable, Category("Filter dimensions")]
        public int FilterWidth
        {
            get { return m_width; }
            set
            {
                if (value < 1)
                    return;
                m_width = value;
                OutputWidth = MyConvolutionLayer.SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, 0);
            }
        }


        [YAXSerializableField(DefaultValue = 2)]
        private int m_height = 2;
        [MyBrowsable, Category("Filter dimensions")]
        public int FilterHeight
        {
            get { return m_height; }
            set
            {
                if (value < 1)
                    return;
                m_height = value;
                OutputHeight = MyConvolutionLayer.SetOutputDimension(InputHeight, FilterHeight, VerticalStride, 0);
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        private int m_horizontalStride = 2;
        [MyBrowsable, Category("Filter dimensions")]
        public int HorizontalStride
        {
            get { return m_horizontalStride; }
            set
            {
                if (value < 1)
                    return;
                m_horizontalStride = value;
                OutputWidth = MyConvolutionLayer.SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, 0);
            }
        }

        [YAXSerializableField(DefaultValue = 2)]
        private int m_verticalStride = 2;
        [MyBrowsable, Category("Filter dimensions")]
        public int VerticalStride
        {
            get { return m_verticalStride; }
            set
            {
                if (value < 1)
                    return;
                m_verticalStride = value;
                OutputHeight = MyConvolutionLayer.SetOutputDimension(InputHeight, FilterHeight, VerticalStride, 0);
            }
        }

        #endregion

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            Neurons = Depth*OutputWidth*OutputHeight;
            OutputColumnHint = OutputWidth;
            base.UpdateMemoryBlocks();
            if (Neurons > 0)
            {
                ActivatedNeurons.Count = Neurons;
                Delta.ColumnHint = OutputWidth;
            }
        }

        //Validation rules
        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (PreviousTopologicalLayer != null)
                validator.AssertError(InputWidth * InputHeight * Depth == PreviousTopologicalLayer.Neurons, this, "'Input width * input height * depth' must be equal to output size of the previous layer.");

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

//    public enum ActivationFunction
//    {
//        MAX
//    }


}
