using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>Generates sample from given distribution</summary>
    /// <description>Uses simple <a href="https://en.wikipedia.org/wiki/Inverse_transform_sampling">Inverse transform sampling</a></description>
    class MyInverseTransformSampling : MyWorkingNode
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                Output.Count = Input.Count;
            }
        }
        
        public override string Description
        {
            get
            {
                return "Distribution";
            }
        }

        public MyITSTask ITS { get; set; }

        /// <description>Implements Inverse transform sampling</description>
        [Description("Inverse transform sampling"), MyTaskInfo(OneShot = false)]
        public class MyITSTask : MyTask<MyInverseTransformSampling>
        {
            private Random m_rnd;
            
            public MyITSTask() { }
            
            public override void Init(int nGPU) 
            {
                m_rnd = new Random();
            }

            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();
                double rnd = m_rnd.NextDouble();
                int idx = 1;
                float sum = Owner.Input.Host[0];
                while (idx < Owner.Input.Count && sum < rnd)
                {
                    sum += Owner.Input.Host[idx];
                    idx++;
                }

                for (int i = 0; i < Owner.Output.Count; i++)
                {
                    Owner.Output.Host[i] = 0;
                }

                Owner.Output.Host[idx - 1] = 1;
                Owner.Output.SafeCopyToDevice();
            }

        }
    }
}
