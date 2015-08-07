using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.SoundProcessing;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.SoundWorld
{
    public class MyAudioPlayerNode: MyWorkingNode, IMyCustomTaskFactory
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [YAXSerializableField]
        protected int m_BufferSize;
        [YAXSerializableField]
        protected int m_BufferSizeSeconds;
        protected WaveFormat m_WaveFormat;

        [Description("Type of input")]
        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 3), YAXElementFor("IO")]
        public int BufferSize
        {
            get { return m_BufferSizeSeconds; }
            set
            {
                m_BufferSizeSeconds = value;
                m_BufferSize = value * SampleRate;
            }
        }

        [Description("Type of input")]
        [MyBrowsable, Category("I/O"), YAXElementFor("IO")]
        [YAXSerializableField(DefaultValue = 44100)]
        public int SampleRate { get; set; }

        private int m_cursor;
        private byte[] m_buffer;
        private WavPlayer m_player;
        private Stream m_stream;
        public MyTask gatherDataTask { get; protected set; }

        public MyAudioPlayerNode() { }

        public override void UpdateMemoryBlocks()
        {
            m_BufferSize = BufferSize * SampleRate;

            if (m_player == null || m_stream == null)
            {
                m_buffer = new byte[m_BufferSize];
                m_stream = new MemoryStream();
                m_WaveFormat = new WaveFormat(SampleRate, 16, 1);
                m_player = new WavPlayer(m_stream, m_WaveFormat, -1, 4096);
            }
        }

        public void CreateTasks()
        {
            gatherDataTask = new MyGatherDataTask();
        }

        [Description("Gather data")]
        class MyGatherDataTask : MyTask<MyAudioPlayerNode>
        {
            // Buffer size should be at least 4096 samples per task execute
            private bool shouldPlay = false;
            private byte[] buffer;

            public override void Init(int nGPU)
            {
                Owner.m_cursor = 0;
                buffer = new byte[Owner.m_BufferSize];
            }

            public override void Execute()
            {
                if (Owner.Input == null)
                    return;

                // Copy data from inputMemBlock, convert it to byte array and write it to the stream
                Owner.Input.SafeCopyToHost();

                // Convert Memory data to bytes
                for (int n = 0; n < Owner.Input.Count; n++)
                {
                    byte[] valueByte = BitConverter.GetBytes((short)Owner.Input.Host[n]);
                    //buffer.AddRange(valueByte);
                    //shouldPlay = buffer.Count % Owner.m_BufferSize == 0 ? true : false;
                    Array.Copy(valueByte, 0, buffer, Owner.m_cursor, valueByte.Length);
                    Owner.m_cursor = (Owner.m_cursor + valueByte.Length);
                    shouldPlay = Owner.m_cursor >= Owner.m_BufferSize? true : false;
                    Owner.m_cursor %= Owner.m_BufferSize;
                }

                if (shouldPlay)
                {
                    Owner.m_stream.Write(buffer, Owner.m_cursor, buffer.Length);
                    Owner.m_stream.Position = 0;
                    Owner.m_player.Play();
                    Owner.m_cursor = 0;
                }
            }
        }
    }
}
