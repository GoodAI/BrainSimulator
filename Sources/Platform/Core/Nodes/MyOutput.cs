using System.Collections.Generic;
using System.Linq;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Utils;

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

        public override bool AcceptsConnection(MyNode fromNode, int fromIndex, int toIndex)
        {
            if (Parent == null)
                return true;

            IEnumerable<MyConnection> connections = Parent.GetConnections(this);
            if (connections == null)
                return true;

            return connections
                .All(connection => connection.To.AcceptsConnection(fromNode, fromIndex, connection.ToIndex));
        }
    }
}
