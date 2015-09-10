using GoodAI.Core.Memory;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Utils;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Core.Observers
{

    public class MyTextObserver : MyAbstractMemoryBlockObserver
    {
        [YAXSerializableField(DefaultValue=16)]
        protected int m_Rows ;

        [YAXSerializableField(DefaultValue = 16)]
        [MyBrowsable, Category("Display"), Description("Maximal number of lines"), DefaultValue(16)]
        public int MaxRows
        {
            get { return m_Rows; }
            set
            {
                if (value >= 0)
                    m_Rows = value;
                else
                    m_Rows = 1;

                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue=80)]
        protected int m_Cols;

        [YAXSerializableField(DefaultValue = 80)]
        [MyBrowsable, Category("Display"), Description("Maximal number of characters per line"), DefaultValue(80)]
        public int MaxLineLength
        {
            get { return m_Cols; }
            set
            {
                if (value >= 0)
                    m_Cols = value;
                else
                    m_Cols = 1;

                TriggerReset();
            }
        }

        [YAXSerializableField]
        private uint BACKGROUND = 0x00;

        private bool isScreenClear = false;
        protected List<String> m_History;
        private CudaDeviceVariable<float> m_HistoryDeviceBuffer;

        private MyCudaKernel m_ClearCanvasKernel;

        /// <summary>
        /// /Constructor with node parameter
        /// </summary>
        public MyTextObserver()
        {
            MaxLineLength = 80;
            MaxRows = 16;

            TextureWidth = 800;
            TextureHeight = 400;

            m_History = new List<string>();
            m_History.Add("");

            m_ClearCanvasKernel = MyKernelFactory.Instance.Kernel(@"GrowingNeuralGas\ClearCanvasKernel");

            TriggerReset();
        }

        /// <summary>
        /// Clear screen kernel
        /// </summary>
        protected void Clear()
        {
            m_ClearCanvasKernel.SetConstantVariable("D_BACKGROUND", BACKGROUND);
            m_ClearCanvasKernel.SetConstantVariable("D_X_PIXELS", TextureWidth);
            m_ClearCanvasKernel.SetConstantVariable("D_Y_PIXELS", TextureHeight);
            m_ClearCanvasKernel.SetupExecution(TextureWidth * TextureHeight);
            m_ClearCanvasKernel.Run(VBODevicePointer);
        }

        protected override void Execute()
        {
            // Clear screen on simulation start
            if(!isScreenClear)
            {
                Clear();
                isScreenClear = true;
            }

            //we are able to represent all characters from ' ' (space) to '~' (tilda) and new-line
            int desiredNum = '~' - ' ' + 2; // the last character is \n

            MyMemoryBlock<float> target = (MyMemoryBlock<float>)Target;
            if(target != null)
            {
                //get data to cpu
                Target.SafeCopyToHost();

                //allow inputs that are different in size, only clamp it if neccessary
                int size = Math.Min(target.Host.Length, desiredNum);

                //find max value for character
                int idx = 0;
                float maxVal = target.Host[0];
                for(int i = 1; i < size; ++i)
                {
                    if(target.Host[idx] < target.Host[i])
                    {
                        idx = i;
                    }
                }

                //reconstruct a character
                char newChar = '\n';
                if (idx + 1 != desiredNum)
                {
                   newChar = (char)(' ' + idx);
                }

                bool splitOccured = false;
                //add character to history but split line it it is too long
                if(newChar == '\n')
                {
                    m_History.Add("");
                }
                else if(m_History[m_History.Count -1].Length >= m_Cols-1)
                {
                    m_History[m_History.Count -1] += "\\";
                    m_History.Add(newChar.ToString());
                    splitOccured = true;
                }
                else
                {
                    m_History[m_History.Count -1] += newChar;
                }

                if(m_History.Count > m_Rows)
                {
                    int rowIdx = 0;
                    //reset gpu data
                    m_HistoryDeviceBuffer.Memset(0);

                    m_History.RemoveAt(0);

                    foreach(string s in m_History)
                    {
                        int colIdx = 0;
                        foreach(char c in s)
                        {
                            m_HistoryDeviceBuffer[m_Cols*rowIdx + colIdx] = (float)(c - ' ');
                            colIdx += 1;
                        }
                        rowIdx += 1;
                    }

                    Clear();

                    for (rowIdx = 0; rowIdx < m_History.Count; ++rowIdx)
                    {
                        MyDrawStringHelper.DrawStringFromGPUMem(m_HistoryDeviceBuffer, 0, rowIdx * (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, rowIdx * m_Cols, m_Cols);
                    }
                }

                else
                {
                    int lastRow = m_History.Count-1;
                    String lastString = m_History[lastRow];
                    if (lastString.Length > 0)
                    {
                        m_HistoryDeviceBuffer[m_Cols * lastRow + lastString.Length - 1] = (float)(lastString.Last() - ' ');
                    }

                    if(splitOccured)
                    {
                        m_HistoryDeviceBuffer[m_Cols * lastRow-1] = (float)('\\' - ' ');
                        MyDrawStringHelper.DrawStringFromGPUMem(m_HistoryDeviceBuffer, 0, (lastRow-1)* (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, (lastRow-1) * m_Cols, m_Cols);
                    }
                    MyDrawStringHelper.DrawStringFromGPUMem(m_HistoryDeviceBuffer, 0, lastRow * (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, lastRow * m_Cols, m_Cols);
                }
            }
        }

        protected override void Reset()
        {
            base.Reset();

            isScreenClear = false;

            // Allocate the history
            m_HistoryDeviceBuffer = new CudaDeviceVariable<float>(m_Rows * m_Cols);
            m_HistoryDeviceBuffer.Memset(0);

            m_History = new List<string>();
            m_History.Add("");
        }
    }
}
