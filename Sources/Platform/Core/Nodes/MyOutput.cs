using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Nodes
{
    public class MyOutput : MyNode
    {        
        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        public override sealed MyMemoryBlock<float> GetOutput(int index)
        {
            return GetInput(index);
        }

        public override sealed MyMemoryBlock<T> GetOutput<T>(int index)
        {
            return GetInput<T>(index);
        }

        public override sealed MyAbstractMemoryBlock GetAbstractOutput(int index)
        {
            return GetAbstractInput(index);
        }
        
        internal MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { }
        }

        public override int OutputBranches
        {
            get { return 0; }
            set { }
        }

        public override void UpdateMemoryBlocks() { }
        public override void Validate(MyValidator validator) { }
    }
}
