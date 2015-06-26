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
using YAXLib;
using ManagedCuda;
using BrainSimulator.Transforms;

namespace BrainSimulator.Motor
{
    /// <author>Karol Kuna</author>
    /// <status>Working</status>
    /// <summary>Records recent input vectors into matrix (one row per time step) and plays the matrix back</summary>
    /// <description></description>
    [YAXSerializeAs("SequenceRecorder")]
    public class MySequenceRecorderNode : MyWorkingNode
    {
        [MyInputBlock]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyInputBlock]
        public MyMemoryBlock<float> PlaybackInput { get { return GetInput(1); } }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> PlaybackOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        public MyMemoryBlock<float> PreviousPlaybackInput { get; set; }
        public MyMemoryBlock<float> IsNewPlaybackInput { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1)]
        public int LENGTH { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1)]
        public int SIZE { get; private set; }

        public MyInitTask InitTask { get; protected set; }
        public MyRecordTask RecordTask { get; protected set; }
        public MyPlaybackTask PlaybackTask { get; protected set; }

        [Description("Init"), MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MySequenceRecorderNode>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.Output.Fill(0.0f);
            }
        }

        [Description("Record"), MyTaskInfo(OneShot = false)]
        public class MyRecordTask : MyTask<MySequenceRecorderNode>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                if (Owner.Input != null)
                {
                    Owner.Output.CopyToMemoryBlock(Owner.Output, 0, Owner.SIZE, (Owner.LENGTH - 1) * Owner.SIZE);
                    Owner.Input.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.SIZE);
                }
            }
        }

        [Description("Playback"), MyTaskInfo(OneShot = false)]
        public class MyPlaybackTask : MyTask<MySequenceRecorderNode>
        {
            private MyCudaKernel m_kernel;

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1)]
            public float PLAYBACK_SPEED { get; set; }

            private float m_playbackIndex;

            public override void Init(int nGPU)
            {
                m_playbackIndex = 0;

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\CompareVectorsKernel");

                if (Owner.PlaybackInput != null)
                {
                    m_kernel.SetupExecution(Owner.SIZE * Owner.LENGTH);
                }
            }

            public override void Execute()
            {
                if (Owner.PlaybackInput != null)
                {
                    Owner.IsNewPlaybackInput.Fill(0);
                    m_kernel.Run(Owner.PlaybackInput, Owner.PreviousPlaybackInput, Owner.IsNewPlaybackInput);

                    Owner.PlaybackInput.SafeCopyToHost();
                    Owner.PreviousPlaybackInput.SafeCopyToHost();

                    Owner.IsNewPlaybackInput.SafeCopyToHost();

                    if (Owner.IsNewPlaybackInput.Host[0] == 0)
                    {
                        m_playbackIndex = Owner.LENGTH - 1;
                    }

                    Owner.PlaybackInput.CopyToMemoryBlock(Owner.PlaybackOutput, ((int) Math.Floor(m_playbackIndex)) * Owner.SIZE, 0, Owner.SIZE);

                    m_playbackIndex -= PLAYBACK_SPEED;

                    if (m_playbackIndex < 0)
                    {
                        m_playbackIndex = Owner.LENGTH - 1;
                    } 
                    else if (m_playbackIndex >= Owner.LENGTH)
                    {
                        m_playbackIndex = 0;
                    }

                    Owner.PlaybackInput.CopyToMemoryBlock(Owner.PreviousPlaybackInput, 0, 0, Owner.SIZE * Owner.LENGTH);
                }
            }
        }


        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                SIZE = Input.Count;
                Output.Count = SIZE * LENGTH;
                Output.ColumnHint = SIZE;
                PlaybackOutput.Count = SIZE;
            }
            if (PlaybackInput != null)
            {
                PreviousPlaybackInput.Count = PlaybackInput.Count;
                IsNewPlaybackInput.Count = 1;
            }
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(Input != null, this, "Input connection missing!");
            validator.AssertError(Input == null || PlaybackInput == null || PlaybackInput.Count == Input.Count * LENGTH, this, "PlaybackInput size must be equal to Input size * LENGTH!");
        }
    }
}
