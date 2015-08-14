using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Resources;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms.Design;
using YAXLib;

namespace GoodAI.Modules.TextProcessing
{
    public class TextWorld : MyWorld
    {
        public enum InputType {Text, Source, UserDefined}
        
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [YAXSerializableField]
        protected string m_Text;

        [YAXSerializableField]
        protected string m_UserDefined;
        
        [YAXSerializableField]
        protected InputType m_UserInput;

        [Description("Path to input text file")]
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("I/O"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string UserDefined 
        {
            get { return m_UserDefined; }
            set
            {
                m_UserDefined = value;

                if(File.Exists(value))
                using (StreamReader sr = new StreamReader(value))
                {
                    m_Text = sr.ReadToEnd();
                }
            } 
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = InputType.Text), YAXElementFor("IO")]
        public InputType UserInput
        {
            get { return m_UserInput; }
            set
            {
                m_UserInput = value;
                if (!(value.CompareTo(InputType.UserDefined) == 0))
                    m_UserDefined = "";

                switch (value)
                {
                    case InputType.Text:
                        m_UserDefined = "Sample Text";
                        m_Text = BasicNodes.Properties.Resources.Wiki;
                        break;
                    case InputType.Source:
                        m_UserDefined = "Sample Source code";
                        m_Text = BasicNodes.Properties.Resources.SourceCode;
                        break;
                    case InputType.UserDefined:
                        m_UserDefined = "";
                        break;
                }
            }
            
        }

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
                if (Owner.m_Text.Length > 0)
                {
                    if (SimulationStep == 0)
                    {
                        Owner.Output.Fill(0);
                    }

                    if (SimulationStep % ExpositionTime == 0)
                    {
                        // convert character into digit index
                        int id = (int)SimulationStep % Owner.m_Text.Length;
                        char c = Owner.m_Text[id];
                        int index = StringToDigitIndexes(c);

                        Array.Clear(Owner.Output.Host, 0, Owner.Output.Count);
                        // if unknown character, continue without setting any connction
                        Owner.Output.Host[index] = 1.00f;
                        Owner.Output.SafeCopyToDevice();
                    }
                }
            }

            public void ExecuteCPU()
            {
                for (int i = 0; i < Owner.UserDefined.Length; i++)
                {
                    char c = Owner.UserDefined[(int)SimulationStep];
                    int index = StringToDigitIndexes(c);

                    Array.Clear(Owner.Output.Host, 0, Owner.Output.Count);
                    // if unknown character, continue without setting any connction
                    Owner.Output.Host[index] = 1.00f;
                }
            }

            private int StringToDigitIndexes(char str)
            {
                int res = 0;
                int charValue = str;
                if (charValue >= ' ' && charValue <= '~')
                    res = charValue - ' ';
                else
                {
                    if (charValue == '\n')
                        res = '~' - ' ' + 1;
                }
                return res;
            }
        }
    }

   
}
