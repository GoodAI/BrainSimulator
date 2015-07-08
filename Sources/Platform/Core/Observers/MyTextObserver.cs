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

        [MyBrowsable, Category("Display"), Description("Maximal number of lines")]
        public int MaxLines
        {
            get
            {
                return m_MaxLines;
            }
            set
            {
                if (value < 0)
                    return;

                m_MaxLines = value;
                TriggerReset();
            }
        }

        protected String m_History;

        private CudaDeviceVariable<float> m_characters; // Reference to the characters bitmaps


        public MyTextObserver() //constructor with node parameter
        {
            MaxLines = 10;
            TextureWidth = 1000;
            TextureHeight = 500;

            m_History = "";
        }

        protected override void Execute()
        {
            int desiredNum = '~' - ' ' + 2; // the last character is \n

            Target.SafeCopyToHost();

            MyMemoryBlock<float> target = (MyMemoryBlock<float>)Target;
            if(target != null)
            {
                //check correct type and size
                int size = Math.Min(target.Host.Length, desiredNum);

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

                if (idx + 1 == desiredNum)
                {
                    m_History += "\n";
                }
                else
                {
                    m_History += (char)(' ' + idx);
                }

                string[] list = m_History.Split('\n');
                int row = 0;
                foreach (string s in list)
                {
                    MyDrawStringHelper.DrawString(s, 0, row * (MyDrawStringHelper.CharacterHeight + 1), 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, 100);
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
