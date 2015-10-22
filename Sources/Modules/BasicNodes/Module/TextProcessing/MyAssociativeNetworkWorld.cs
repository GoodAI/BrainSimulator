using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using System.Windows.Forms.Design;
using System.Drawing.Design;
using System.IO;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using System.Text.RegularExpressions;
using GoodAI.Core.Signals;




namespace GoodAI.Modules.LTM
{
    /// <author>GoodAI</author>
    /// <meta>pd, jv</meta>
    /// <status>working</status>
    /// <summary> World outputting concepts and relations loaded from a file or typed in by the user. </summary>
    /// <description>
    /// World outputting concepts and relations (as strings encoded into numeric vectors, one letter one value (ASCII value - 32)) loaded from a file or typed in by the user. 
    /// The information is stored in a comma separated format: <b>A, B, R, p</b> on each line, where <b>A</b> and <b>B</b> are words, <b>R</b> is the relation <b>A->B</b> and <b>p</b> is it's strength. 
    /// 
    /// <h3>output Memory Blocks</h3>
    /// <ul>
    ///     <li> <b>Concept1:</b> Concept <b>A</b> coded as a float vector of integer values corresponding to each letter.</li>
    ///     <li> <b>Concept2:</b> Concept <b>B</b> coded as a float vector of integer values corresponding to each letter.</li>
    ///     <li> <b>Relation:</b> Relation <b>R</b> coded as a float vector of integer values corresponding to each letter.</li>
    ///     <li> <b>RealtionStrength:</b> The strength <b>p</b> of the relation <b>R</b> (<b>A->B</b>)</li>
    /// </ul>
    /// 
    /// 
    /// <h3>Signals</h3>
    /// <ul>
    ///     <li> <b>MyDatabaseReadInProgress:</b> signal is raised during first pass of the dataset.</li>
    ///     <li> <b>MyDatabaseReadFinished:</b> signal is raised for one time step, when the first row of the dataset is read for the second time.</li>
    /// </ul>
    /// </description>
    class MyAssociativeNetworkWorld : MyWorld
    {
        #region Memory Blocks
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Concept1
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Concept2
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> Relation
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> RelationStrength
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        #endregion

        #region Properties

        private int m_readingFrequency = 1;
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1), Description("New relation is read every ReadingFrequency ticks.")]
        public int ReadingFrequency
        { 
            get { return m_readingFrequency; }
            set
            {
                if (value > 0)
                {
                    m_readingFrequency = value;
                }
            }


        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 20), Description("Maximum width of each element (maximum number of characters). If the element string is shorter, space characters will be added. If longer, it will be truncated.")]
        public int TextWidth { get; set; }

        [MyBrowsable, Category("Params"), DefaultValue("concept 1, concept 2, relation, 0.9")]
        [YAXSerializableField(DefaultValue = "concept 1, concept 2, relation, 0.9"), YAXElementFor("IO")]
        public string UserText
        {
            get;
            set;

        }

        [Description("Path to input text file")]
        [YAXSerializableField(DefaultValue = "associations.txt"), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("Params"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string UserFile
        {
            get;
            set;
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = UserInput.UserFile), YAXElementFor("IO")]
        public UserInput InputType
        {
            get;
            set;
        }
        public enum UserInput { UserText, UserFile }



        #endregion

        #region Variables

        protected string m_text;

        protected string[] m_parsedText;



        public class MyDatabaseReadFinishedSignal : MySignal { }
        public MyDatabaseReadFinishedSignal ReadFinishedSignal { get; private set; }

        public class MyDatabaseReadInProgress : MySignal { }
        public MyDatabaseReadInProgress ReadInProgressSignal { get; private set; }

        #endregion


        public MyReadInputTask ReadInput { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Concept1.Count = TextWidth;
            Concept2.Count = TextWidth;
            Relation.Count = TextWidth;
            RelationStrength.Count = 1;

            Concept1.ColumnHint = TextWidth;
            Concept2.ColumnHint = TextWidth;
            Relation.ColumnHint = TextWidth;

        }


        /// <summary>
        /// Reads the whole input file / user text and then outputs the relations one at a time. New relations are outputted every <b>ReadingFrequency</b> ticks.
        /// </summary>
        [Description("Read relations")]
        public class MyReadInputTask : MyTask<MyAssociativeNetworkWorld>
        {
            public override void Init(int nGPU)
            {

                switch (Owner.InputType)
                {
                    case UserInput.UserText:
                        MyLog.DEBUG.WriteLine("Reading input for MyAssociativeWorld... from UserText.");
                        Owner.m_text = Owner.UserText;
                        break;
                    case UserInput.UserFile:
                        MyLog.DEBUG.WriteLine("Reading input for MyAssociativeWorld... from file " + Owner.UserFile + ".");
                        Owner.m_text = "";

                        if (File.Exists(Owner.UserFile))
                        {
                            using (StreamReader sr = new StreamReader(Owner.UserFile))
                            {
                                String[] output = sr.ReadToEnd().Split(Environment.NewLine.ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
                                Owner.m_text = String.Join("\n", output);
                            }
                        }
                        else
                        {
                            MyLog.ERROR.WriteLine("Cannot read file " + Owner.UserFile);
                        }
                        break;
                }
                Owner.m_parsedText = null;
                if (Owner.m_text.Length > 0)
                {
                    Owner.m_parsedText = Regex.Split(Owner.m_text, "\r\n|\r|\n");
                }
                Owner.ReadFinishedSignal.Drop();
                Owner.ReadInProgressSignal.Raise();
            }
            
            public override void Execute()
            {
                // ReadFinishedSignal should be active only one time step
                if (Owner.ReadFinishedSignal.IsRised())
                {
                    Owner.ReadFinishedSignal.Drop();
                }

                if (SimulationStep % Owner.ReadingFrequency == 0)
                {
                    long id = SimulationStep / Owner.ReadingFrequency;

                    if (Owner.m_parsedText == null)
                    {
                        MyLog.ERROR.WriteLine("No input data for MyAssociationWorld!");
                        return;
                    }

                    // let others know that reading has finished
                    if (id == Owner.m_parsedText.Length)
                    {
                        Owner.ReadInProgressSignal.Drop();
                        Owner.ReadFinishedSignal.Raise();
                    }

                    id = id % Owner.m_parsedText.Length;

                    //   MyLog.DEBUG.WriteLine("Id = " + id + " / " + Owner.m_parsedText.Length);
                    // MyLog.DEBUG.WriteLine("str = " + Owner.m_parsedText[id]);

                    String[] elements = Regex.Split(Owner.m_parsedText[id], @"\s*,\s*"); //spliting expression is any number of white spaces, comma, and again any number of white spaces

                    if (elements.Length < 3 || elements.Length > 4)
                    {
                        string message;
                        if (elements.Length < 3)
                            message = "few";
                        else
                            message = "many";

                        MyLog.WARNING.WriteLine("Too " + message + " (" + elements.Length + " instead of 3 or 4) elements in input line '" + Owner.m_parsedText[id] + "' for " + Owner.Name + ". Output will be empty.");
                        return;

                    }


                    MyLog.DEBUG.WriteLine("Id = " + id + " / " + Owner.m_parsedText.Length);
                    MyLog.DEBUG.WriteLine("conc1 = " + elements[0]);
                    MyLog.DEBUG.WriteLine("conc2 = " + elements[1]);
                    MyLog.DEBUG.WriteLine("relation = " + elements[2]);

                    if (elements.Length == 3) //input file does not contain relation strength
                    {
                        Owner.RelationStrength.Host[0] = 1.0f;
                    }
                    else
                    {
                        try
                        {
                            Owner.RelationStrength.Host[0] = float.Parse(elements[3]);
                        }
                        catch
                        {
                            MyLog.WARNING.WriteLine(Owner.Name + " expects the fourth element in the input line '" + Owner.m_parsedText[id] + "' to be a number.");
                            Owner.RelationStrength.Host[0] = float.NaN;
                        }

                    }
                    MyLog.DEBUG.WriteLine("relation strength = " + Owner.RelationStrength.Host[0]);


                    String elem1 = elements[0].PadRight(Owner.TextWidth);
                    String elem2 = elements[1].PadRight(Owner.TextWidth);
                    String rel = elements[2].PadRight(Owner.TextWidth);


                    for (int i = 0; i < Owner.TextWidth; i++)
                    {
                        Owner.Concept1.Host[i] = MyStringConversionsClass.StringToDigitIndexes(elem1[i]);
                        Owner.Concept2.Host[i] = MyStringConversionsClass.StringToDigitIndexes(elem2[i]);
                        Owner.Relation.Host[i] = MyStringConversionsClass.StringToDigitIndexes(rel[i]);
                    }

                    Owner.Concept1.SafeCopyToDevice();
                    Owner.Concept2.SafeCopyToDevice();
                    Owner.Relation.SafeCopyToDevice();

                    // MyLog.DEBUG.WriteLine("CCC_" + concept + "_CCC");

                    Owner.RelationStrength.SafeCopyToDevice();
                }
            }
        }
    }
}
