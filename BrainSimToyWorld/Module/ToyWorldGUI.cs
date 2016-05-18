using System.Linq;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.ToyWorldAPI;

namespace GoodAI.ToyWorld
{
    public class ToyWorldGUI : MyWorkingNode
    {
        public TWGUITask GUITask { get; private set; }

        public event MessageEventHandler MessageObtained = delegate { };
        public event MessageEventHandler StringObtained = delegate { };

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> MessageOut
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> StringOut
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> MessageIn
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> StringIn
        {
            get { return GetInput(1); }
        }

        private int m_messageSize;
        private string m_message;

        public string Message
        {
            get { return m_message; }
            set { m_message = Truncate(value, m_messageSize); }
        }

        public string String { get; set; }

        public override void UpdateMemoryBlocks()
        {
            m_messageSize = MessageOut.Count = MessageIn.Count;
            StringOut.Count = StringIn.Count;
        }

        private string Truncate(string value, int length)
        {
            if (string.IsNullOrEmpty(value)) return value;
            return value.Length <= length ? value : value.Substring(0, length);
        }

        [MyTaskInfo(OneShot = false)]
        public class TWGUITask : MyTask<ToyWorldGUI>
        {
            public override void Init(int nGPU) { }

            public override void Execute()
            {
                ProcessMessage();
                ProcessString();
            }

            private void ProcessString()
            {
                Owner.StringIn.CopyToMemoryBlock(Owner.StringOut, 0, 0, Owner.StringIn.Count);
                Owner.String = string.Join("", Owner.StringIn.Host.Select(x => (char)x));
                Owner.StringObtained(Owner, new MessageEventArgs(Owner.String));
            }

            private void ProcessMessage()
            {
                if (Owner.Message == null)
                {
                    Owner.MessageIn.SafeCopyToHost();
                    string text = string.Join("", Owner.MessageIn.Host.Select(x => (char)x));
                    Owner.MessageObtained(Owner, new MessageEventArgs(text));

                    Owner.MessageIn.CopyToMemoryBlock(Owner.MessageOut, 0, 0, Owner.m_messageSize);
                }
                else
                {
                    for (int i = 0; i < Owner.Message.Length; ++i)
                        Owner.MessageOut.Host[i] = Owner.Message[i];
                    Owner.MessageOut.SafeCopyToDevice();

                    Owner.Message = null;
                }
            }
        }
    }
}
