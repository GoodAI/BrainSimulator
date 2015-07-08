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
        [YAXSerializableField]
        protected int m_MaxLines;

        [MyBrowsable, Category("Display"), Description("Maximal number of lines"), DefaultValue(10)]
        public int MaxLines
        {
            get { return m_MaxLines; }
            set
            {
                if (value >= 0)
                    m_MaxLines = value;
                else
                    m_MaxLines = 0;

                TriggerReset();
            }
        }

        protected String m_History;

        private CudaDeviceVariable<float> m_characters; // Reference to the characters bitmaps


        public MyTextObserver() //constructor with node parameter
        {
            m_History = "";
            TextureWidth = 800;
            TextureHeight = 400;
        }

        protected override void Execute()
        {
            int endOfLine = '~' - ' ' + 2; // the last character is \n

            Target.SafeCopyToHost();

            MyMemoryBlock<float> target = (MyMemoryBlock<float>)Target;
            if(target != null)
            {
                //check correct type and size
                int size = Math.Min(target.Host.Length, endOfLine);

                //find max value
                int idx = 0;
                float maxVal = target.Host[0];

                for(int i = 1; i < size; ++i)
                {
                    if(target.Host[idx] < target.Host[i])
                    {
                        idx = i;
                    }
                }

                if (idx + 1 == endOfLine)
                {
                    m_History += "\n";
                }
                else
                {
                    m_History += (char)(' ' + idx);
                }

                int row = 0;
                string[] list = m_History.Split('\n');
                foreach (string s in list)
                {
                    MyDrawStringHelper.DrawString(s, 0, row * (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, 80);
                    row += 1;
                }
            }
        }

        protected override void Reset()
        {
            base.Reset();

            m_History = "";
        }
    }
}
