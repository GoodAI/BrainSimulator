using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Drawing.Design;
using System.IO;
using System.Text;
using System.Globalization;
using System.Windows.Forms.Design;
using GoodAI.Core.Execution;
using YAXLib;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>js</meta>
    /// <status>Working</status>
    /// <summary>Node for generating CSV files. </summary>
    /// <description>The generated file starts with (possibly several lines of) headers
    /// copied from the property "Headers". Then every other line contains consecutively:
    /// the time step, the label and input value (formatted according to the property
    /// "InputValueWriteFormat" and separated by the property "ListSeparator").
    /// One line corresponds to one data point.
    /// </description>
    public class MyCsvFileWriterNode : MyWorkingNode
    {
        public enum FileWriteMethod
        {
            Overwrite,
            Append
        }

        public enum ValueWriteFormat
        {
            // the input array is transformed into string containing only zeros and ones
            // input value lower than threshold (0.5) becomes 0, larger than threshold becomes 1
            BinaryString,
            // the input array is written on one line as values separated by the "ListSeparator" property
            Array
        }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = "outputDirectory"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(FolderNameEditor), typeof(UITypeEditor))]
        public string OutputDirectory { get; set; }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = "outputFile.csv"), YAXElementFor("Structure")]
        public string OutputFile { get; set; }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = FileWriteMethod.Overwrite)]
        public FileWriteMethod WriteMethod
        {
            get { return m_writeMethod; }
            set
            {
                m_writeMethod = value;
                // If this changed, we need to restart the appending logic. See OnSimulationStateChanged.
                m_isAlreadyWriting = false;
            }
        }
        private FileWriteMethod m_writeMethod;

        [MyBrowsable, Category("Content")]
        [YAXSerializableField, YAXElementFor("Structure")]
        [EditorAttribute(typeof(MultilineStringEditor), typeof(UITypeEditor))]
        public string Headers { get; set; }

        [MyBrowsable, Category("Content")]
        [YAXSerializableField(DefaultValue = ValueWriteFormat.BinaryString), YAXElementFor("Structure")]
        public ValueWriteFormat InputValueWriteFormat { get; set; }

        [MyBrowsable, Category("Content")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool IncludeTimeStep { get; set; }

        [MyBrowsable, Category("Content")]
        [YAXSerializableField(DefaultValue = true), YAXElementFor("Structure")]
        public bool IncludeLabel { get; set; }

        [MyBrowsable, Category("Content")]
        [YAXSerializableField, YAXElementFor("Structure")]
        [EditorAttribute(typeof(MultilineStringEditor), typeof(UITypeEditor))]
        public string ListSeparator
        {
            get { return listSeparator; }
            set
            {
                if (String.IsNullOrEmpty(value))
                    listSeparator = ",";
                else
                    listSeparator = value;
            }
        }
        private string listSeparator;
        protected StreamWriter Stream;

        // If the writer is in overwrite mode, it must still append when the simulation is paused and resumed.
        // This flag indicates that the file has been written into in this simulation run and therefore
        // it has to be open in append mode again.
        private bool m_isAlreadyWriting;

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputSize { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputWidth { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputHeight { get; private set; }

        public MyCsvFileWriterNode()
        {
            listSeparator = CultureInfo.CurrentCulture.TextInfo.ListSeparator;
            Headers = "timestamp" + listSeparator + "label" + listSeparator + "data";
        }


        // INPUT / OUTPUT -----------------------------------------------------
        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> Label
        {
            get { return GetInput(1); }
        }

        // MEMORY BLOCKS ----------------------------------------------------
        public MyMemoryBlock<int> ActiveCellsIndices { get; private set; }

        public MyWriterTask SpTask { get; protected set; }

        public override void OnSimulationStateChanged(MySimulationHandler.StateEventArgs args)
        {
            base.OnSimulationStateChanged(args);

            if (args.NewState == MySimulationHandler.SimulationState.RUNNING ||
                args.NewState == MySimulationHandler.SimulationState.RUNNING_STEP)
            {
                InitStream();
            }
            else
            {
                // Simulation stopped? => set the flag to false so that we use the selected writing mode when it starts again.
                if (args.NewState == MySimulationHandler.SimulationState.STOPPED)
                    m_isAlreadyWriting = false;

                CloseStream();
            }
        }

        private void InitStream()
        {
            if (Stream != null)
                return;

            MyLog.DEBUG.WriteLine("Opening the CSV file handle");


            // If the file was supposed to be overwritten, but that already happened in this simulation, append.
            // Otherwise, it would overwrite the file after each "pause" or simulation restart.
            // This is basically done to keep the original functionality even though we reopen the file now.
            bool append = m_isAlreadyWriting || WriteMethod == FileWriteMethod.Append;
            m_isAlreadyWriting = true;
            Stream = new StreamWriter(OutputDirectory + '\\' + OutputFile, append);
        }

        private void CloseStream()
        {
            if (Stream == null)
                return;

            Stream.Close();
            Stream = null;
        }


        // TASKS -----------------------------------------------------------
        [Description("Write Row"), MyTaskInfo(OneShot = false)]
        public class MyWriterTask : MyTask<MyCsvFileWriterNode>
        {
            int m_step;

            public override void Init(int nGPU)
            {
                m_step = 0;

                Owner.InitStream();

                // when appending, dont add the headers
                if (!String.IsNullOrEmpty(Owner.Headers) && (Owner.WriteMethod == FileWriteMethod.Overwrite))
                {
                    StringBuilder sb = new StringBuilder();
                    if (Owner.Headers.EndsWith(Environment.NewLine))
                    {
                        Owner.Headers.Remove(Owner.Headers.Length - 1);
                    }
                    sb.Append(Owner.Headers);
                    Owner.Stream.WriteLine(sb);
                }
            }

            public override void Execute()
            {
                if ((Owner.Input != null) && (Owner.Label != null))
                {
                    Owner.Input.SafeCopyToHost();
                    Owner.Label.SafeCopyToHost();
                    StringBuilder sb = new StringBuilder();
                    if (Owner.IncludeTimeStep)
                    {
                        sb.Append(m_step);
                        sb.Append(Owner.ListSeparator);
                    }
                    if (Owner.IncludeLabel)
                    {
                        sb.Append(Owner.Label.Host[0].ToString("0.00000"));
                        sb.Append(Owner.ListSeparator);
                    }

                    if (Owner.InputValueWriteFormat == ValueWriteFormat.BinaryString)
                    {
                        for (int i = 0; i < Owner.Input.Count; i++)
                        {
                            if (Owner.Input.Host[i] < 0.5)
                            {
                                sb.Append('0');
                            }
                            else
                            {
                                sb.Append('1');
                            }
                        }
                    }
                    else if (Owner.InputValueWriteFormat == ValueWriteFormat.Array)
                    {
                        for (int i = 0; i < Owner.Input.Count; i++)
                        {
                            sb.Append(Owner.Input.Host[i].ToString("0.00"));
                            sb.Append(Owner.ListSeparator);
                        }
                    }
                    Owner.Stream.WriteLine(sb.ToString());
                }
                m_step++;
            }
        }

        public override void UpdateMemoryBlocks()
        {
            InputSize = Input == null ? 1 : (uint)Input.Count;
            InputWidth = Input == null ? 1 : (uint)Input.ColumnHint;
            InputHeight = (uint)Math.Ceiling((float)InputSize / (float)InputWidth);
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(Directory.Exists(OutputDirectory), this, "The output directory does not exist.");
        }

    }
}
