using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.LTM
{
    /// <author>GoodAI</author>
    /// <meta>pd,vkl</meta>
    /// <status>working</status>
    /// <summary> Node for inputing a line of text. </summary>
    /// <description>
    /// Converts the input number into a string. If there is no input, converts the user defined Text. Output is encoded as a vector of integers.
    /// 
    /// <h3>input Memory Blocks</h3>
    /// <ul>
    ///     <li> <b>InputNumber:</b> Number which should be converted into a float vector of integer values corresponding to each letter. </li>
    /// </ul>
    /// <h3>output Memory Blocks</h3>
    /// <ul>
    ///     <li> <b>Output:</b> The number taken from <b>Input</b> or in the case of no input the text from property variable <b>Text</b> coded as a float vector of integer values corresponding to each letter. It can be optionally transformed to upper case letters. </li>
    /// </ul>
    /// 
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li> <b>TextWidth:</b> Width of the output (maximum number of characters). If the input is shorter, space characters will be added. If longer, it will be truncated. </li>
    /// </ul>
    /// </description>
    class MyTextInputNode : MyWorkingNode
    {

        #region MemoryBlocks
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> InputNumber
        {
            get { return GetInput(0); }
        }

        #endregion

        #region Parameters

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 20), Description("Width of the output (maximum number of characters). If the input is shorter, space characters will be added. If longer, it will be truncated.")]
        public int TextWidth { get; set; }

        #endregion


        public override void UpdateMemoryBlocks()
        {
            Output.Count = TextWidth;
            Output.ColumnHint = Output.Count;
        }

        public MyOutputTextTask WriteOutput { get; private set; }

        public override void Validate(MyValidator validator)
        {

            //either there is no input, or the input shoud be a scalar
            if (InputNumber != null)
            {
                validator.AssertWarning(InputNumber.Count == 1, this, "There should be no input or the input should be a scalar. (Using just the first element out of " + InputNumber.Count + ".)");
            }
           


        }




        /// <summary>
        /// Converts the input number into a string. If there is no input, converts the user defined Text. Output is encoded as a vector of integers.
        /// </summary>
        [Description("Output text")]
        public class MyOutputTextTask : MyTask<MyTextInputNode>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = "User text"), DefaultValue("User text"), Description("String to appear on the output.")]
            public String Text { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = false), DefaultValue(false), Description("Enables automatic conversion to capital letters.")]
            public bool ConvertToUpperCase { get; set; }
            
            [Description("Use ASCII-based Digit Index encoding (default) or Uppercase Vowel-Space-Consonant encoding")]
            [MyBrowsable, Category("Encoding")]
            [YAXSerializableField(DefaultValue = MyStringConversionsClass.StringEncodings.DigitIndexes)]
            public MyStringConversionsClass.StringEncodings Encoding { get; set; }

            [Description("How should the input be padded?")]
            [MyBrowsable, Category("Encoding")]
            [YAXSerializableField(DefaultValue = MyStringConversionsClass.PaddingSchemes.None)]
            public MyStringConversionsClass.PaddingSchemes Padding { get; set; }
            
            public override void Init(int nGPU)
            {
                
                      
            }

            public override void Execute()
            {
                String word;
                if (Owner.InputNumber != null)
                {
                    Owner.InputNumber.SafeCopyToHost();
                    word = Owner.InputNumber.Host[0].ToString("F8");
                }
                else
                {
                    word = Text;
                }

               


                String padded = word.PadRight(Owner.TextWidth);

                if (ConvertToUpperCase)
                {
                    padded = padded.ToUpper();
                    //MyLog.DEBUG.WriteLine("text: " + padded);
                }

                if (Padding == MyStringConversionsClass.PaddingSchemes.Repeat)
                {
                    padded = MyStringConversionsClass.RepeatWord(padded, Owner.TextWidth);
                }
                else if (Padding == MyStringConversionsClass.PaddingSchemes.Stretch)
                {
                    padded = MyStringConversionsClass.StretchWord(padded, Owner.TextWidth);
                }

                
                for (int i = 0; i < Owner.TextWidth; i++)
                {
                    if (Encoding == MyStringConversionsClass.StringEncodings.DigitIndexes)
                    {
                        Owner.Output.Host[i] = MyStringConversionsClass.StringToDigitIndexes(padded[i]);
                    }
                    else
                    {
                        Owner.Output.Host[i] = MyStringConversionsClass.StringToUvscCoding(padded[i]);
                    }
                }

                Owner.Output.SafeCopyToDevice();
            }
        }
    }
}
