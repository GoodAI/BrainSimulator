using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using BrainSimulator.Memory;
using System.Drawing;
using YAXLib;
using ManagedCuda;


namespace BrainSimulator.Motor
{
    /// <author>Karol Kuna</author>
    /// <status>Working</status>
    /// <summary>Transforms column from stators output in SEDroneWorld to vector</summary>
    /// <description></description>
    [YAXSerializeAs("ColumnToVector")]
    public class MyColumnToVectorNode : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("Column")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int COLUMN { get; set; }

        public MyConvertColumnToVectorTask ColumnToVector { get; protected set; }

        [Description("Convert column to vector"), MyTaskInfo(OneShot = false, Order = 0)]
        public class MyConvertColumnToVectorTask : MyTask<MyColumnToVectorNode>
        {
            public override void Init(int nGPU)
            {
                
            }

            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();

                int columns = Owner.Input.ColumnHint;
                int rows = Owner.Input.Count / Owner.Input.ColumnHint;

                for (int i = 0; i < rows; i++)
                {
                    Owner.Output.Host[i] = Owner.Input.Host[i * columns + Owner.COLUMN];
                }

                Owner.Output.SafeCopyToDevice();
            }
        }

        public MyConvertColumnToRotorRotationTask RotorRotation { get; protected set; }

        [Description("Rotor Rotation"), MyTaskInfo(OneShot = false, Order = 0)]
        public class MyConvertColumnToRotorRotationTask : MyTask<MyColumnToVectorNode>
        {
            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();

                int columns = Owner.Input.ColumnHint;
                int rows = Owner.Input.Count / Owner.Input.ColumnHint;

                for (int i = 0; i < rows; i++)
                {
                    float minRotation = Owner.Input.Host[i * columns + 0];
                    float maxRotation = Owner.Input.Host[i * columns + 1];

                    float interval = maxRotation - minRotation;
                    float currentRotation = Owner.Input.Host[i * columns + 5];

                    if (interval == 0.0f)
                    {
                        Owner.Output.Host[i] = 0.0f;
                    }
                    else
                    {
                        Owner.Output.Host[i] = (currentRotation - minRotation) / interval;
                    }
                }

                Owner.Output.SafeCopyToDevice();
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                Output.Count = Input.Count / Input.ColumnHint;
            }
        }
    }
}
