using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.IO;
using System.Windows.Forms.Design;
using YAXLib;

namespace GoodAI.Modules.TextProcessing
{
    /// <author>GoodAI</author>
    /// <meta>mh</meta>
    /// <status>Working</status>
    /// <summary>Provides sample or custom text input for additional processing.</summary>
    /// <description>Provides sample or custom text input for additional processing.</description>
    public class TextWorld : MyWorld
    {
        public enum UserInput {UserText, UserFile}
        
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        protected string m_Text;

        [YAXSerializableField]
        protected string m_UserFile;
        
        [YAXSerializableField]
        protected UserInput m_UserInput;

        public MyCUDAGenerateInputTask GenerateInput { get; protected set; }

        #region I/O
        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = "User text"), YAXElementFor("IO")]
        public string UserText { get; set; }

        [Description("Path to input text file")]
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("I/O"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string UserFile 
        {
            get { return m_UserFile; }
            set
            {
                InputType = UserInput.UserFile;
                m_UserFile = value;
            } 
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = UserInput.UserText), YAXElementFor("IO")]
        public UserInput InputType
        {
            get { return m_UserInput; }
            set
            {
                m_UserInput = value;
            }
        }
        #endregion

        // Parameterless constructor
        public TextWorld() { }

        public override void UpdateMemoryBlocks()
        {
            //we are able to represent all characters from ' ' (space) to '~' (tilda) and new-line(/n)
            Output.Count = '~'-' ' + 2; 
        }

        /// <summary>Provides sample or custom text input for additional processing.</summary>
        [Description("Read text inputs")]
        public class MyCUDAGenerateInputTask : MyTask<TextWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public int ExpositionTime { get; set; }

            public override void Init(Int32 nGPU)
            {
                //read file/user-input
                switch (Owner.InputType)
                {
                    case UserInput.UserText:
                        Owner.m_Text = Owner.UserText;
                        break;
                    case UserInput.UserFile:
                        if (File.Exists(Owner.UserFile))
                            using (StreamReader sr = new StreamReader(Owner.UserFile))
                            {
                                Owner.m_Text = sr.ReadToEnd();
                            }
                        break;
                }
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
                for (int i = 0; i < Owner.m_Text.Length; i++)
                {
                    char c = Owner.m_Text[(int)SimulationStep];
                    int index = StringToDigitIndexes(c);

                    Array.Clear(Owner.Output.Host, 0, Owner.Output.Count);
                    // if unknown character, continue without setting any connction
                    Owner.Output.Host[index] = 1.00f;
                }
            }

            /// <summary>
            /// Converts char to index in ASCII table.
            /// </summary>
            /// <param name="str">Input char.</param>
            /// <returns>Index of char in ASCII table.</returns>
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
