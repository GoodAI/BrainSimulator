using System;
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
    /// <summary>Convolutional layer.</summary>
    /// <description>
    /// Classic convolutional layer that performs convolution on image windows using its filters.\n
    /// Great tutorial on filter and input dimensions is available here: http://cs231n.github.io/convolutional-networks/ \n \n
    /// 
    /// You can use AutomaticInput to determine parameters of the convolution. By default, it will try to preserve input dimensions on the output.
    /// 
    /// 
    ///  </description>
    public class MyConvolutionLayer : MyAbstractWeightLayer, IMyCustomTaskFactory
    {

        public MyConvolutionInitLayerTask InitLayerTask { get; protected set; }
        public MyConvolutionUpdateWeights UpdateWeights { get; protected set; }


        #region Parameters

        public override ConnectionType Connection
        {
            get { return ConnectionType.CONVOLUTION; }
        }


        [YAXSerializableField(DefaultValue = 128)]
        [MyBrowsable, Category("\tLayer"), ReadOnly(true)]
        public override int Neurons { get; set; }



        #region Input
        [MyBrowsable, Category("Input image dimensions"), DisplayName("\t\tWidth"), ReadOnly(true)]
        public int InputWidth
        {
            get { return GetInputDimension(Input, 0); }

        }


        [MyBrowsable, Category("Input image dimensions"), DisplayName("\tHeight"), ReadOnly(true)]
        public int InputHeight
        {
            get { return GetInputDimension(Input, 1); }

        }


        [MyBrowsable, Category("Input image dimensions"), DisplayName("Depth"), ReadOnly(true)]
        public int InputDepth
        {
            get { return GetInputDimension(Input, 2); }

        }
        #endregion


        #region Output
        [MyBrowsable, Category("Output image dimensions"), DisplayName("\t\tWidth"), ReadOnly(true)]
        public int OutputWidth
        {
            get { return SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, ZeroPadding); }
        }

        [MyBrowsable, Category("Output image dimensions"), DisplayName("\tHeight"), ReadOnly(true)]
        public int OutputHeight
        {
            get { return SetOutputDimension(InputHeight, FilterHeight, VerticalStride, ZeroPadding); }
        }

        [MyBrowsable, Category("Output image dimensions"), DisplayName("Depth"), ReadOnly(true)]
        public int OutputDepth
        {
            get { return FilterCount; }
        }
        #endregion


        #region Filter
        [YAXSerializableField(DefaultValue = 3)]
        private int m_width = 3;
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

        [YAXSerializableField(DefaultValue = 3)]
        private int m_height = 3;
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

        [MyBrowsable, Category("Filter"), DisplayName("\tDepth"), ReadOnly(true)]
        public int FilterDepth
        {
            get { return InputDepth; }
        }

        [YAXSerializableField(DefaultValue = 16)]
        private int m_filterCount = 16;
        [MyBrowsable, Category("Filter"), DisplayName("Count")]
        public int FilterCount
        {
            get {return m_filterCount;}
            set
            {
                if (value < 1)
                    return;
                m_filterCount = value;
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_horizontalStride = 1;
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

        [YAXSerializableField(DefaultValue = 1)]
        private int m_verticalStride = 1;
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

        [YAXSerializableField(DefaultValue = 1)]
        private int m_zeroPadding = 1;
        [MyBrowsable, Category("Filter")]
        public int ZeroPadding
        {
            get { return m_zeroPadding; }
            set
            {
                if (value < 0)
                    return;
                m_zeroPadding = value;
            }
        }


        [YAXSerializableField(DefaultValue = false)]
        private bool m_autoOutput = false;
        [MyBrowsable, Category("Filter"), DisplayName("ZeroPadding automatic")]
        public bool AutomaticOutput
        {
            get { return m_autoOutput; }
            set
            {
                m_autoOutput = false;
                if (value)
                {
                    int currentOutSize = 1 + (int)Math.Ceiling((InputWidth - FilterWidth) / (double)HorizontalStride);
                    if (((InputWidth - currentOutSize) % 2) != 0)
                        MyLog.WARNING.WriteLine("Zero padding of " + this.Name + " set automatically but filter will not cover the whole padded image when convolving.");
                    else
                    {
                        MyLog.INFO.WriteLine("Output parameters of layer " + this.Name + " were set succesfully.");
                    }
                    ZeroPadding = (int)Math.Ceiling((InputWidth - currentOutSize) / (double)2);

                }
            }
        }
        #endregion



        #endregion

        #region Memory blocks
        public MyMemoryBlock<float> PaddedImage { get; protected set; }


        #endregion


        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            
            Neurons = OutputWidth*OutputHeight*FilterCount;

            base.UpdateMemoryBlocks();

            if (Neurons > 0)
            {
                Output.Dims = new TensorDimensions(OutputWidth, OutputHeight, OutputDepth);
                // allocate memory scaling with number of neurons in layer
                Delta.Count = Neurons;
                Delta.Dims = Output.Dims;
                Bias.Count = FilterCount;
                PreviousBiasDelta.Count = Neurons; // momentum method

                // RMSProp allocations
                MeanSquareWeight.Count = Weights.Count;
                MeanSquareBias.Count = Bias.Count;


                // allocate memory scaling with input
                if (Input != null)
                {
                    PaddedImage.Count = InputDepth * (InputWidth + 2 * ZeroPadding) * (InputHeight + 2 * ZeroPadding);
                    PaddedImage.Dims = new TensorDimensions(OutputWidth + 2 * ZeroPadding, OutputHeight + 2 * ZeroPadding, InputDepth);

                    Weights.Count = FilterWidth * FilterHeight * InputDepth * FilterCount;
                    Weights.Dims = new TensorDimensions(FilterWidth, FilterHeight, InputDepth, FilterCount);

                    PreviousWeightDelta.Count = Weights.Count;
                    PreviousWeightDelta.Dims = Weights.Dims;

                    MeanSquareWeight.Dims = Weights.Dims;

                    NeuronInput.Dims = Input.Dims;
                }

                if (Weights.Count % 2 != 0)
                    Weights.Count++;
                if (Bias.Count % 2 != 0)
                    Bias.Count++;


            }
        }

        //Validation rules
        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            validator.AssertError((InputHeight - FilterHeight + 2 * ZeroPadding) % VerticalStride == 0, this, "Filter doesn't fit vertically when striding.");

            validator.AssertError((InputWidth - FilterWidth + 2 * ZeroPadding) % HorizontalStride == 0, this, "Filter doesn't fit horizontally when striding.");

            validator.AssertInfo(ZeroPadding == (FilterWidth - 1) / 2 && ZeroPadding == (FilterHeight - 1) / 2, this, "Input and output might not have the same dimension. Set stride to 1 and zero padding to ((FilterSize - 1) / 2) to fix this.");
        }

        // description
        public override string Description
        {
            get
            {
                return "Convolutional layer";
            }
        }


        public void CreateTasks()
        {
            ForwardTask = new MyConvolutionForwardTask();
            DeltaBackTask = new MyConvolutionBackwardTask();
        }



        public static int GetInputDimension(MyMemoryBlock<float> memoryBlock, int dimension)
        {
            if (memoryBlock == null)
                return 0;

            return dimension >= memoryBlock.Dims.Count ? 1 : memoryBlock.Dims[dimension];

        }

        public static int SetOutputDimension(int inputSize, int filterSize, int stride, int zeroPadding = 0)
        {
            return 1 + (zeroPadding * 2 + inputSize - filterSize) / stride;
        }
    }
}
