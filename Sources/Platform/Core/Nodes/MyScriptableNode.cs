using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public abstract class MyScriptableNode : MyWorkingNode
    {
        [YAXSerializableField(DefaultValue = "")]
        public string Script { get; set; }
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

        public MyTestingScriptableNode()
        {
            Script = "Some testing code;\r\nSome more lines;";
        }

        public override void UpdateMemoryBlocks()
        {
            
        }
    }  
}
