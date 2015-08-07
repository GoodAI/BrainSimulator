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

namespace GoodAI.SoundWorld
{
    public class MyAudioPlayerNode: MyWorkingNode, IMyCustomTaskFactory
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        private int m_cursor;
        private WavPlayer m_player;
        private Stream m_stream;
        public MyTask gatherDataTask { get; protected set; }

        public MyAudioPlayerNode() { }

        public override void UpdateMemoryBlocks()
        {
            if (m_player == null || m_stream == null)
            {
                m_stream = new MemoryStream();
                m_player = new WavPlayer(m_stream, new WaveFormat(44100, 16, 2),-1, 4096);
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
            private int buff_size = 8192;
            private byte[] buffer;
            private bool shouldPlay = false;

            public override void Init(int nGPU) 
            {
                Owner.m_cursor = 0;
                buff_size = Math.Max(8192, Owner.Input.Count);
                buffer = new byte[buff_size];
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
                    byte[] valueByte = BitConverter.GetBytes(Owner.Input.Host[n]);
                    Array.Copy(valueByte, 0, buffer, Owner.m_cursor, valueByte.Length);
                    Owner.m_cursor = (Owner.m_cursor + valueByte.Length);
                    shouldPlay = Owner.m_cursor >= buff_size ? true : false;
                    Owner.m_cursor %= buff_size;
                }
                

                if (shouldPlay)
                {
                    Owner.m_stream.Write(buffer, Owner.m_cursor, buff_size);
                    Owner.m_stream.Position = 0;
                    Owner.m_player.Play();
                    Owner.m_cursor = 0;
                }
            }
        }

    }
}
