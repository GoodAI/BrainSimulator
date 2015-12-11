using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Design;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms.Design;
using YAXLib;


namespace GoodAI.Modules.LTM
{
    /// <author>GoodAI</author>
    /// <meta>pd,os,vkl</meta>
    /// <status>working</status>
    /// <summary>
    ///    Node for visualizing vectors as words.
    /// </summary>
    /// <description>
    /// 
    /// The sole purpose of this node is to gather a matrix and visualize each of it's vectors as words (i.e. chains of characters). This visualization is performed via a custom observer - MyVectorTextObserver.
    /// <h3>Input Memory Blocks</h3>
    /// <ul>
    ///     <li> <b>Data:</b> Data matrix (M x N) containing one word in each row. In each element is stored one character as an integer depending on the chosen encoding.</li>
    ///     <li> <b>Weights:</b> A vector (of size M) describing importance of each word. The words are written in a descending order according to this weights and the weights also determine the brightness of each word.</li>
    /// </ul>
    /// </description>
    public class MyTextObserverNode : MyWorkingNode
    {


        [MyInputBlock(0)]
        public MyMemoryBlock<float> Data
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Weights
        {
            get { return GetInput(1); }
        }


       
        public MyWriteToFileTask WriteFile { get; private set; }

        [Description("Use ASCII-based Digit Index encoding (default) or Uppercase Vowel-Space-Consonant encoding")]
        [MyBrowsable, Category("Encoding")]
        [YAXSerializableField(DefaultValue = MyStringConversionsClass.StringEncodings.DigitIndexes)]
        public MyStringConversionsClass.StringEncodings Encoding { get; set; }

        public override void UpdateMemoryBlocks()
        {
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            validator.AssertError(Data.Count / Data.ColumnHint == Weights.Count, this, "There must be the same number of rows in Data (" + Data.Count / Data.ColumnHint + ") as number of elements in Weights vector (" + Weights.Count + ").");


        }

    }


    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>
    /// Extracts text from the vector and writes it on File. This task is currently supposed to be used in conjunction with LTM, where each column is a concatenation of one or more vectors, on vectors that represent text.
    /// The extracted text can be formatted accordingly to represent Concept1,Concept2,Relationship,ConceptStrength,RelationStrength (In such case SplitValue should be 5, value 1 means don't split)
    /// </summary>
    /// <description></description>
    [Description("Write on File"), MyTaskInfo(Disabled = true)]
    public class MyWriteToFileTask : MyTask<MyTextObserverNode>
    {

        //Allow user to specify path for writing on file
        [Description("Path to text file to write to")]
        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("FileWriting"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string WriteOnFilePath
        {
            get;
            set;
        }


        [MyBrowsable, Category("FileWriting"), YAXSerializableField(DefaultValue = false),
        Description("If true, prints the relevant output also in the Debug section. ")]
        public bool ShowDebugInfo
        {
            get;
            set;
        }

        [MyBrowsable, Category("FileWriting"), Description("The vector can be the result of a concatenation of one or more vectors, SplitValue specifies in how many parts to split, (1 means don't split)")]
        [YAXSerializableField(DefaultValue = 5)]
        public int SplitValue { get; set; }


        public override void Init(int nGPU)
        {

        }

        public override void Execute()
        {

            if (WriteOnFilePath == null || WriteOnFilePath == "")
            {
                MyLog.WARNING.WriteLine(Owner.Name + ": The Task 'Write On File' is selected, but the specified path for writing on File (FileWritePath) is empty, not writing on file");
            }
            else
            {

                Owner.Data.SafeCopyToHost();
                Owner.Weights.SafeCopyToHost();

                string ReturnDataSplit = "";
                string ReturnData = "";

                int NumberOfVectors = SplitValue;
                int SplitRange = Owner.Data.ColumnHint / NumberOfVectors;
                int VectorLength = Owner.Data.ColumnHint;

                if (ShowDebugInfo)
                {
                    MyLog.DEBUG.WriteLine("Data Length: " + Owner.Data.Host.Length);
                    MyLog.DEBUG.WriteLine("Vector Length: " + VectorLength);
                    MyLog.DEBUG.WriteLine("Number of Vectors: " + NumberOfVectors);
                    MyLog.DEBUG.WriteLine("SplitRange: " + SplitRange);
                    MyLog.DEBUG.WriteLine("Number of entries: " + Owner.Data.Host.Length / Owner.Data.ColumnHint);
                    MyLog.DEBUG.WriteLine("Weights Length: " + Owner.Weights.Host.Length);
                }

                int consecutiveSpaces = 0;
                int spaceChar;

                if (Owner.Encoding == MyStringConversionsClass.StringEncodings.UVSC)
                    spaceChar = MyStringConversionsClass.UvscCodingToDigitIndexes(' ');
                else
                    spaceChar = MyStringConversionsClass.StringToDigitIndexes(' ');

                StringBuilder sb = new StringBuilder();

                for (int z = 0; z < Owner.Data.Host.Length; z++)
                {
                    if (Owner.Data.Host[z] == spaceChar)
                    {
                        consecutiveSpaces++;
                        //end the writing prematurely if there is an empty line (for printing long buffers filled with just a few lines)
                        if (consecutiveSpaces > Owner.Data.ColumnHint)
                        {
                            if (z % Owner.Data.ColumnHint == spaceChar)
                            {
                                break;
                            }
                        }
                    }
                    else
                    {
                        consecutiveSpaces = 0;
                    }

                    if (Owner.Encoding == MyStringConversionsClass.StringEncodings.DigitIndexes)
                        sb.Append(MyStringConversionsClass.DigitIndexToString(Owner.Data.Host[z]));
                    else
                        sb.Append(MyStringConversionsClass.UvscCodingToString(Owner.Data.Host[z]));
                }


                ReturnData = sb.ToString();
                string current = "";
                String[] elements;
                StringBuilder sbSplit = new StringBuilder();


                for (int j = 0; j < ReturnData.Length; j += VectorLength)
                {
                    sbSplit.Append(Math.Round(Owner.Weights.Host[j / VectorLength], 2) + " "); //Round the weight to 2 decimals places

                    for (int n = 0; n < NumberOfVectors; n++)
                    {
                        current = ReturnData.Substring(((j) + (n * (SplitRange))), SplitRange); //Divide chunk depending on the number of vectors
                        //MyLog.DEBUG.WriteLine("Analyzing chunk " + n + ", Substring with index " + ((j) + (n * (SplitRange))) + " with length  " + SplitRange + " = " + current);
                        elements = Regex.Split(current, @"\s*,\s*");                            //Spliting expression is any number of white spaces, comma, and again any number of white spaces
                        elements[0] = elements[0].Trim();                                       //Trim formats the text so that there are no spaces from the word/s and the beginning and the end of the length
                        sbSplit.Append(elements[0]);

                        if (n != (NumberOfVectors - 1))
                        {
                            sbSplit.Append(",");
                        }
                    }
                    
                    sbSplit.Append(Environment.NewLine);
                }

                ReturnDataSplit = sbSplit.ToString();

                //Convert string to stream
                byte[] byteArray = Encoding.UTF8.GetBytes(ReturnDataSplit);
                MemoryStream stream = new MemoryStream(byteArray);
                StreamReader reader = new StreamReader(stream);

                //Before writing on file, delete specific lines from the stream, the lines containing the string ",,", because they contain incomplete information
                string search_text = ",,";
                string old;
                StringBuilder sbNo = new StringBuilder();
                StreamReader sr = reader;
                int DeletedLines = 0;

                while ((old = sr.ReadLine()) != null)
                {

                    if (!old.Contains(search_text))
                    {
                        sbNo.Append(old);
                        sbNo.Append(Environment.NewLine);
                    }
                    else
                    {
                        DeletedLines++;
                    }

                }

                sr.Close();
                string FormattedResult = sbNo.ToString();
                File.WriteAllText(WriteOnFilePath, FormattedResult);

                if (ShowDebugInfo)
                {
                    MyLog.DEBUG.WriteLine("OutPut: " + Environment.NewLine + FormattedResult);
                    MyLog.DEBUG.WriteLine("Number of lines that were deleted during the process because they contained the charachter ',,' = " + DeletedLines);
                }

            }
        }

    }





}
