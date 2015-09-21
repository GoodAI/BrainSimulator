using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public abstract class MyScriptableNode : MyWorkingNode
    {
        [YAXSerializableField]
        protected string m_script;

        protected event EventHandler<MyPropertyChangedEventArgs<string>> ScriptChanged;

        public string Script 
        {
            get { return m_script; }
            set 
            {
                string oldValue = m_script;
                m_script = value;
                if (ScriptChanged != null)
                {
                    ScriptChanged(this, new MyPropertyChangedEventArgs<string>(oldValue, m_script));
                }
            }
        }
    }

    public class MyTestingScriptableNode : MyScriptableNode 
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }        

        public override void UpdateMemoryBlocks()
        {
            
        }
    }  
}
