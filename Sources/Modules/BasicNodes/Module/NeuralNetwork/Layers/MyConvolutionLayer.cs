using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using CustomModels.NeuralNetwork.Tasks;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Layers
{



    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>WIP</status>
    /// <summary>Convolutional layer.</summary>
    /// <description></description>
    public class MyConvolutionLayer : MyAbstractWeightLayer, IMyCustomTaskFactory
    {

        public MyPadImageTask PadImageTask { get; protected set; }
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



        [YAXSerializableField(DefaultValue = 16)]
        [MyBrowsable, Category("Filter")]
        public int FilterCount { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_inputDepth = 1;
        [MyBrowsable, Category("Input image dimensions")]
        public int InputDepth
        {
            get { return m_inputDepth; }
            set
            {
                if (value < 1)
                    return;
                m_inputDepth = value;
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
                OutputWidth = SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, ZeroPadding);
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
                OutputHeight = SetOutputDimension(InputHeight, FilterHeight, VerticalStride, ZeroPadding);
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

        [YAXSerializableField(DefaultValue = 1)]
        private int m_zeroPadding = 1;
        [MyBrowsable, Category("Output image dimensions")]
        public int ZeroPadding
        {
            get { return m_zeroPadding; }
            set
            {
                if (value < 0)
                    return;
                m_zeroPadding = value;
                OutputWidth = SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, ZeroPadding);
                OutputHeight = SetOutputDimension(InputHeight, FilterHeight, VerticalStride, ZeroPadding);
            }
        }



        [YAXSerializableField(DefaultValue = 3)]
        private int m_width = 3;
        [MyBrowsable, Category("Filter")]
        public int FilterWidth
        {
            get { return m_width; }
            set
            {
                if (value < 1)
                    return;
                m_width = value;
                OutputWidth = SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, ZeroPadding);
            }
        }

        [YAXSerializableField(DefaultValue = 3)]
        private int m_height = 3;
        [MyBrowsable, Category("Filter")]
        public int FilterHeight
        {
            get { return m_height; }
            set
            {
                if (value < 1)
                    return;
                m_height = value;
                OutputHeight = SetOutputDimension(InputHeight, FilterHeight, VerticalStride, ZeroPadding);
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_horizontalStride = 1;
        [MyBrowsable, Category("Filter")]
        public int HorizontalStride
        {
            get { return m_horizontalStride; }
            set
            {
                if (value < 1)
                    return;
                m_horizontalStride = value;
                OutputWidth = SetOutputDimension(InputWidth, FilterWidth, HorizontalStride, ZeroPadding);
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_verticalStride = 1;
        [MyBrowsable, Category("Filter")]
        public int VerticalStride
        {
            get { return m_verticalStride; }
            set
            {
                if (value < 1)
                    return;
                m_verticalStride = value;
                OutputHeight = SetOutputDimension(InputHeight, FilterHeight, VerticalStride, ZeroPadding);
            }
        }



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
                // allocate memory scaling with number of neurons in layer
                Delta.Count = Neurons;
                Delta.ColumnHint = OutputWidth;
                Bias.Count = FilterCount;
                PreviousBiasDelta.Count = Neurons; // momentum method

                // RMSProp allocations
                MeanSquareWeight.Count = Weights.Count;
                MeanSquareBias.Count = Bias.Count;

                // Adadelta allocation
                AdadeltaWeight.Count = Weights.Count;
                AdadeltaBias.Count = Bias.Count;

                if (ZeroPadding > 0)
                {
                    PaddedImage.Count = InputDepth*(InputWidth + 2*ZeroPadding)*(InputHeight + 2*ZeroPadding);
                    PaddedImage.ColumnHint = InputWidth + 2*ZeroPadding;
                }
                else
                    PaddedImage.Count = 0;

                // allocate memory scaling with input
                if (Input != null)
                {
                    Weights.Count = FilterWidth * FilterHeight * InputDepth * FilterCount;
                    Weights.ColumnHint = FilterWidth;

                    PreviousWeightDelta.Count = Weights.Count;
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

        public static int SetOutputDimension(int inputSize, int filterSize, int stride, int zeroPadding)
        {
            return 1 + (zeroPadding * 2 + inputSize - filterSize) / stride;
        }


        public void CreateTasks()
        {
            ForwardTask = new MyConvolutionForwardTask();
            DeltaBackTask = new MyConvolutionBackwardTask();
        }

    }
}
