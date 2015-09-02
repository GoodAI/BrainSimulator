using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Records sequence of input vectors into a matrix</summary>
    /// <description>Records sequence of predefined number of recent Inputs and stores them in Output as a matrix. Most recent input vector is in first row of the matrix.<br />
    ///              Parameters:
    ///              <ul>
    ///                 <li>LENGTH: Length of the sequence to record, also number of rows in the output matrix</li>
    ///              </ul>
    /// </description>
    [YAXSerializeAs("SequenceRecorder")]
    public class MySequenceRecorderNode : MyWorkingNode
    {
        [MyInputBlock]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1)]
        public int LENGTH { get; set; }

        public MyRecordTask RecordTask { get; protected set; }

        /// <summary>Records the sequence</summary>
        [Description("Record"), MyTaskInfo(OneShot = false)]
        public class MyRecordTask : MyTask<MySequenceRecorderNode>
        {
            public override void Init(int nGPU) {}

            public override void Execute()
            {
                if (SimulationStep == 0)
                {
                    Owner.Output.Fill(0);
                }
                Owner.Output.CopyToMemoryBlock(Owner.Output, 0, Owner.Input.Count, (Owner.LENGTH - 1) * Owner.Input.Count);
                Owner.Input.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.Input.Count);
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                Output.Count = Input.Count * LENGTH;
                Output.ColumnHint = Input.Count;
            }
        }

        public override string Description
        {
            get
            {
                return "SequenceRecorder";
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
        }
    }
}
