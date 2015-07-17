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

namespace GoodAI.Core.Observers
{

    public class MyTextObserver : MyAbstractMemoryBlockObserver
    {
        [YAXSerializableField, DefaultValue(10)]
        protected int m_Rows;

        [MyBrowsable, Category("Display"), Description("Maximal number of lines"), DefaultValue(10)]
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

        [YAXSerializableField, DefaultValue(10)]
        protected int m_Cols;

        [MyBrowsable, Category("Display"), Description("Maximal number of characters per line"), DefaultValue(10)]
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
        private uint BACKGROUND = 0xFFFFFFFF;

        protected List<String> m_History;
        private CudaDeviceVariable<int> m_HistoryDeviceBuffer;

        private MyCudaKernel m_ClearCanvasKernel;


        public MyTextObserver() //constructor with node parameter
        {
            TextureWidth = 800;
            TextureHeight = 400;

            m_History = new List<string>();
            m_History.Add("");

            m_ClearCanvasKernel = MyKernelFactory.Instance.Kernel(@"GrowingNeuralGas\ClearCanvasKernel");
        }

        protected override void Execute()
        {
            int maxRowLen = 80;
            int maxRows = 16;

            //clear kernel
            m_ClearCanvasKernel.SetConstantVariable("D_BACKGROUND", BACKGROUND);
            m_ClearCanvasKernel.SetConstantVariable("D_X_PIXELS", TextureWidth);
            m_ClearCanvasKernel.SetConstantVariable("D_Y_PIXELS", TextureHeight);
            m_ClearCanvasKernel.SetupExecution(TextureWidth * TextureHeight);
            m_ClearCanvasKernel.Run(VBODevicePointer);

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

                //add character to history but split line it it is too long
                if(newChar == '\n')
                {
                    m_History.Add("");
                }
                else if(m_History[m_History.Count -1].Length >= maxRowLen)
                {
                    m_History[m_History.Count -1] += "\\";
                    m_History.Add(newChar.ToString());
                }
                else
                {
                    m_History[m_History.Count -1] += newChar;
                }

                int row = 0;
       
                //print only last maxRows lines
                foreach (string s in m_History.GetRange(Math.Max(0, m_History.Count - maxRows), Math.Min(maxRows, m_History.Count)))
                {
                    if (s.Length != 0)
                    {
                        MyDrawStringHelper.DrawStringFromGPUMem(m_HistoryDeviceBuffer, 0, row * (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight);
                    }
                    row += 1;
                }
            }
        }

        protected override void Reset()
        {
            base.Reset();

            // Allocate the history
            m_HistoryDeviceBuffer = new CudaDeviceVariable<int>(m_Rows * m_Cols);
            m_HistoryDeviceBuffer.Memset(1);

            m_History = new List<string>();
            m_History.Add("");
        }
    }
}
