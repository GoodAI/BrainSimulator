using BEPUphysics;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>3D World based on BEPUphysics engine.
    /// It can simulate various constraint systems defined via XML.</summary>
    /// <description></description>
    public class My3DWorld : MyWorld
    {
        public Space Space { get; set; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Controls
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Joints
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public override void UpdateMemoryBlocks() {}

        public override void Cleanup()
        {
            Space = null;
        }
    }
}
