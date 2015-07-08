using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.TextProcessing
{
    public class TextWorld : MyWorld
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = "aa bb cc"), YAXElementFor("IO")]
        public string UserInput { get; set; }

        public MyCUDAGenerateInputTask GenerateInput { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = '~'-' ' + 2; // last character is \n
        }

        [Description("Read text inputs")]
        public class MyCUDAGenerateInputTask : MyTask<TextWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public int ExpositionTime { get; set; }

            public override void Init(Int32 nGPU)
            {
                // do nothing here
            }

            public override void Execute()
            {
                if (Owner.UserInput.Length > 0)
                {
                    if (SimulationStep == 0)
                    {
                        Owner.Output.Fill(0);
                    }

                    if (SimulationStep % ExpositionTime == 0)
                    {
                        // convert character into digit index
                        int id = (int)SimulationStep % Owner.UserInput.Length;
                        char c = Owner.UserInput[id];
                        int index = StringToDigitIndexes(c);

                        Array.Clear(Owner.Output.Host, 0, Owner.Output.Count);
                        // if unknown character, continue without setting any connction
                        if (index > -1)
                        {
                            Owner.Output.Host[index] = 1.00f;
                            Owner.Output.SafeCopyToDevice();
                        }
                    }
                }
            }

            public void ExecuteCPU()
            {
                for (int i = 0; i < Owner.UserInput.Length; i++)
                {
                    char c = Owner.UserInput[(int)SimulationStep];
                    int index = StringToDigitIndexes(c);

                    Array.Clear(Owner.Output.Host, 0, Owner.Output.Count);
                    // if unknown character, continue without setting any connction
                    if (index > -1)
                    {
                        Owner.Output.Host[index] = 1.00f;
                    }
                }
            }

            private int StringToDigitIndexes(char str)
            {
                int res = -1;
                    int charValue = str;
                    if (charValue >= ' ' && charValue <= '~')
                        res = charValue - ' ';
                    else
                    {
                        if (charValue == '\n')
                            res = '~' + 1;
                    }
                return res;
            }
        }
    }

   
}
