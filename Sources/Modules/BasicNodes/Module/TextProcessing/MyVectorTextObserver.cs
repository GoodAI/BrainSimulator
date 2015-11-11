using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers.Helper;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Core.Task;
using GoodAI.Core;

namespace GoodAI.Modules.LTM
{

    /// <author>GoodAI</author>
    /// <meta>pd,vkl</meta>
    /// <status>working</status>
    /// <summary>
    ///    Custom observer for visualizing words - works on MyTextObserverNode.
    /// </summary>
    /// <description>
    /// See MyTextObserverNode for more details.
    /// </description>
    class MyVectorTextObserver : MyNodeObserver<MyTextObserverNode>
    {


        #region Constants
        private const float DEFAULT_upperThreshold = 0.75f;
        private const float DEFAULT_lowerThreshold = 0.5f;
        private const int ROW_HEIGHT = 15;
        private const int ROW_WIDTH = 9;
        private const uint BACKGROUND = 0x00;
        private const int DEFAULT_maxWordSize = 20;
        private const int DEFAULT_maxRows = 20;

        #endregion

        #region Properties

        [YAXSerializableField(DefaultValue = DEFAULT_maxRows)]
        protected int m_Rows;

        [YAXSerializableField(DefaultValue = DEFAULT_maxRows), MyBrowsable, Category("Display"), Description("Maximal number of lines"), DefaultValue(DEFAULT_maxRows)]
        public int MaxRows
        {
            get { return m_Rows; }
            set
            {
                if (value >= 0)
                {
                    m_Rows = value;
                }
                else
                {
                    m_Rows = 1;
                }
                UpdateTextureSizes();
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = DEFAULT_maxWordSize)]
        protected int m_maxWordSize;

        [YAXSerializableField(DefaultValue = DEFAULT_maxWordSize)]
        [MyBrowsable, Category("Display"), Description("Maximal number of characters per line"), DefaultValue(DEFAULT_maxWordSize)]
        public int MaxWordSize
        {
            get { return m_maxWordSize; }
            set
            {
                if (value >= 0)
                {
                    m_maxWordSize = value;
                }
                else
                {
                    m_maxWordSize = 1;
                }
                UpdateCols();
            }
        }

        [YAXSerializableField(DefaultValue = DEFAULT_upperThreshold)]
        private float m_upperThreshold = DEFAULT_upperThreshold;

        private float m_lowestWeight;
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 0.001f), Description("The minimum accepted weight, all weights will have at least this weight (for drawing).")]
        public float DarkestWeight
        {
            get { return m_lowestWeight; }
            set { m_lowestWeight = value; }
        }

        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = DEFAULT_upperThreshold), Description("Concepts with similarity over this threshold will be displayed using the brightest color.")]
        public float UpperThreshold
        {
            get { return m_upperThreshold; }
            set { m_upperThreshold = value; }
        }

        [YAXSerializableField(DefaultValue = DEFAULT_lowerThreshold)]
        private float m_lowerThreshold = DEFAULT_lowerThreshold;

        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = DEFAULT_lowerThreshold), Description("Concepts with similarity below this threshold will not be displayed at all.")]
        public float LowerThreshold
        {
            get { return m_lowerThreshold; }
            set { m_lowerThreshold = value; }
        }

        [YAXSerializableField(DefaultValue = false)]
        private bool m_printWeights = false;

        [MyBrowsable, Category("Display"), YAXSerializableField(DefaultValue = false), Description("Turns on printing of weights."), DefaultValue(false)]
        public bool PrintWeights
        {
            get { return m_printWeights; }
            set
            {
                m_printWeights = value;
                UpdateCols();
            }
        }

        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = true), Description("Turns on sorting data by weights."), DefaultValue(true)]
        public bool SortByWeights
        {
            get;
            set;
        }

        #endregion

        #region Private Variables

        [YAXSerializableField(DefaultValue = DEFAULT_maxWordSize)]
        private int m_cols;



        private CudaDeviceVariable<float> m_deviceBuffer;

        private MyCudaKernel m_clearCanvasKernel;



        #endregion


        private void UpdateTextureSizes()
        {
            TextureWidth = m_cols * ROW_WIDTH;
            TextureHeight = m_Rows * ROW_HEIGHT;
        }


        private void UpdateCols()
        {
            m_cols = m_maxWordSize;
            if (m_printWeights)
            {
                m_cols += 5;
            }
            UpdateTextureSizes();
            TriggerReset();
        }

        /// <summary>
        /// /Constructor with node parameter
        /// </summary>
        public MyVectorTextObserver()
        {
            MaxWordSize = DEFAULT_maxWordSize;
            MaxRows = DEFAULT_maxRows;

            UpdateTextureSizes();


            m_clearCanvasKernel = MyKernelFactory.Instance.Kernel(@"GrowingNeuralGas\ClearCanvasKernel");

            TriggerReset();
        }

        /// <summary>
        /// Clear screen kernel
        /// </summary>
        protected void Clear()
        {
            m_clearCanvasKernel.SetConstantVariable("D_BACKGROUND", BACKGROUND);
            m_clearCanvasKernel.SetConstantVariable("D_X_PIXELS", TextureWidth);
            m_clearCanvasKernel.SetConstantVariable("D_Y_PIXELS", TextureHeight);
            m_clearCanvasKernel.SetupExecution(TextureWidth * TextureHeight);
            m_clearCanvasKernel.Run(VBODevicePointer);
        }

        protected override void Execute()
        {

            if (m_lowerThreshold >= m_upperThreshold)
            {
                MyLog.ERROR.WriteLine("MyVectorTextObserver - LowerThreshold needs to be lower than UpperThreshold.");
            }


            //clear screen
            Clear();


           
            Target.Data.SafeCopyToHost();
            Target.Weights.SafeCopyToHost();



          

            float[] weights = Target.Weights.Host;
            int[] indexes = new int[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                indexes[i] = i;
            }

            if (SortByWeights)
            {
                Array.Sort(Target.Weights.Host, indexes);
                Array.Reverse(Target.Weights.Host);
                Array.Reverse(indexes);
            }


            int m_numberOfRows = Target.Data.Count / Target.Data.ColumnHint;
            if (m_numberOfRows > m_Rows)
            {
                m_numberOfRows = m_Rows;
            }
            if (m_numberOfRows > Target.Weights.Count)
            {
                m_numberOfRows = Target.Weights.Count;
            }



            bool displayedWarning = false;
            for (int i = 0; i < m_numberOfRows; i++)
            {
                if (Target.Weights.Host[i] <= m_lowerThreshold)
                {
                    continue;
                }

                int WeightsStringLength = 0;
                if (m_printWeights) //weights have to be printed in front of the concepts
                {
                    string WeigthsString = Target.Weights.Host[i].ToString("F2") + " ";
                    WeightsStringLength = WeigthsString.Length;

                    for (int j = 0; j < WeightsStringLength; j++)
                    {
                        m_deviceBuffer[i * m_cols + j] = MyStringConversionsClass.StringToDigitIndexes(WeigthsString[j]);
                    }

                    // MyLog.DEBUG.WriteLine("wText : " + WeigthsString);
                }


                int numberOfColumns = Target.Data.ColumnHint;
                int windowWidth = m_cols - WeightsStringLength;
                if (numberOfColumns > windowWidth)
                {
                    if (!displayedWarning)
                    {
                        MyLog.WARNING.WriteLine("Text lines (length " + numberOfColumns + ") cannot fit into MyVectorTextObserver window width (" + windowWidth + "), they will be cropped.");
                        displayedWarning = true;
                    }
                    numberOfColumns = windowWidth;

                }


                for (int j = 0; j < numberOfColumns; j++)
                {
                    if (Target.Encoding == MyStringConversionsClass.StringEncodings.DigitIndexes)
                    {
                        m_deviceBuffer[i * m_cols + j + WeightsStringLength] = Target.Data.Host[indexes[i] * Target.Data.ColumnHint + j];
                    }
                    else
                    {
                        m_deviceBuffer[i * m_cols + j + WeightsStringLength] = MyStringConversionsClass.UvscCodingToDigitIndexes(
                            Target.Data.Host[indexes[i] * Target.Data.ColumnHint + j]);
                    }
                }

                MyDrawStringHelper.DrawStringFromGPUMem(m_deviceBuffer, 0, i * (MyDrawStringHelper.CharacterHeight + 1), 0, ComputeColor(Target.Weights.Host[i]), VBODevicePointer, TextureWidth, TextureHeight, i * m_cols, m_cols);
            }
        }

        uint ComputeColor(float weight)
        {
            weight = (weight - LowerThreshold) / (UpperThreshold - LowerThreshold);


            if (weight < 0)
            {
                weight = 0;
            }
            if (weight > 1)
            {
                weight = 1.0f;
            }
            if (weight < m_lowestWeight)
            {
                weight = m_lowestWeight;
            }
            uint color = 0x000100 * (uint)Math.Round(weight * 255);

            return color;

        }

        protected override void Reset()
        {
            base.Reset();



            // Allocate the buffer
            m_deviceBuffer = new CudaDeviceVariable<float>(m_Rows * m_cols);
            m_deviceBuffer.Memset(0);
        }
    }
}
