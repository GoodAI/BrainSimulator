using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using System.Diagnostics;
using YAXLib;

namespace GoodAI.Core.Nodes
{    
    public class MyParentInput : MyNode
    {
        public MyParentInput()
        {
            base.OutputBranches = 1;
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { }
        }

        [YAXSerializableField, YAXAttributeForClass]
        public int ParentInputIndex { get; internal set; }        

        public override sealed MyMemoryBlock<float> GetOutput(int index)
        {
            Debug.Assert(index == 0, "ParentInput cannot have multiple outputs");
            return Parent != null ? Parent.GetInput(ParentInputIndex) : null;
        }

        public override sealed MyMemoryBlock<T> GetOutput<T>(int index)
        {
            Debug.Assert(index == 0, "ParentInput cannot have multiple outputs");
            return Parent != null ? Parent.GetInput<T>(ParentInputIndex) : null;
        }

        public override MyAbstractMemoryBlock GetAbstractOutput(int index)
        {
            Debug.Assert(index == 0, "ParentInput cannot have multiple outputs");
            return Parent != null ? Parent.GetAbstractInput(ParentInputIndex) : null;
        }

        public override int OutputBranches
        {
            get { return base.OutputBranches; }
            set { }
        }

        public override void UpdateMemoryBlocks() { }
        public override void Validate(MyValidator validator) { }
    }
}
