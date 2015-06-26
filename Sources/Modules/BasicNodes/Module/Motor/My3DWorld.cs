using BEPUphysics;
using BEPUphysics.BroadPhaseEntries.MobileCollidables;
using BEPUphysics.CollisionRuleManagement;
using BEPUphysics.CollisionShapes;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysics.Constraints.SolverGroups;
using BEPUphysics.Constraints.TwoEntity.Motors;
using BEPUphysics.Entities;
using BEPUphysics.Entities.Prefabs;
using BEPUphysics.Materials;
using BEPUutilities;
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.Motor
{
    /// <author>Dusan Fedorcak</author>
    /// <status>WIP</status>
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
