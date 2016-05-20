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
        [MyTaskGroup("TWGUITask")]
        public TWGUIDisplayTask DisplayTask { get; private set; }
        [MyTaskGroup("TWGUITask")]
        public TWGUIInterceptTask InterceptTask { get; private set; }

        public event MessageEventHandler TextObtained = delegate { };

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> TextOut
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> TextIn
        {
            get { return GetInput(0); }
        }

        private int m_messageSize;
        private string m_text;

        public string Text
        {
            get { return m_text; }
            set { m_text = Truncate(value, m_messageSize); }
        }

        // I don't want to create new enum just for two values
        public bool IsDisplay
        {
            get { return DisplayTask.Enabled; }
        }

        public bool IsIntercept
        {
            get { return InterceptTask.Enabled; }
        }

        public override void UpdateMemoryBlocks()
        {
            if (TextIn != null)
                m_messageSize = TextOut.Count = TextIn.Count;
        }

        private string Truncate(string value, int length)
        {
            if (string.IsNullOrEmpty(value)) return value;
            return value.Length <= length ? value : value.Substring(0, length);
        }

        [MyTaskInfo(OneShot = false)]
        public class TWGUIDisplayTask : MyTask<ToyWorldGUI>
        {
            public override void Init(int nGPU) { }

            public override void Execute()
            {
                Owner.TextIn.SafeCopyToHost();
                Owner.TextIn.CopyToMemoryBlock(Owner.TextOut, 0, 0, Owner.TextIn.Count);
                Owner.Text = string.Join("", Owner.TextIn.Host.Select(x => (char)x));
                Owner.TextObtained(Owner, new MessageEventArgs(Owner.Text, "Display"));
            }
        }

        [MyTaskInfo(OneShot = false)]
        public class TWGUIInterceptTask : MyTask<ToyWorldGUI>
        {
            public override void Init(int nGPU) { }

            public override void Execute()
            {
                if (Owner.Text == null)
                {
                    Owner.TextIn.SafeCopyToHost();
                    string text = string.Join("", Owner.TextIn.Host.Select(x => (char)x));
                    Owner.TextObtained(Owner, new MessageEventArgs(text, "Intercept"));

                    Owner.TextIn.CopyToMemoryBlock(Owner.TextOut, 0, 0, Owner.m_messageSize);
                }
                else
                {
                    for (int i = 0; i < Owner.Text.Length; ++i)
                        Owner.TextOut.Host[i] = Owner.Text[i];
                    Owner.TextOut.SafeCopyToDevice();

                    Owner.Text = null;
                }
            }
        }
    }
}
