using System;
using System.ComponentModel;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;

namespace GoodAI.School.Worlds
{
    [DisplayName("ToyWorld")]
    public class ToyWorldAdapterWorld : ToyWorld.ToyWorld, IWorldAdapter
    {
        public SchoolWorld School { get; set; }
        public MyWorkingNode World { get { return this; } }

        #region Implementation of IWorldAdapter

        public MyTask GetWorldRenderTask()
        {
            return UpdateTask;
        }

        public override MyMemoryBlock<float> GetInput(int index)
        {
            return School.GetInput(index);
        }

        public override MyMemoryBlock<T> GetInput<T>(int index)
        {
            return School.GetInput<T>(index);
        }

        public override MyAbstractMemoryBlock GetAbstractInput(int index)
        {
            return School.GetAbstractInput(index);
        }

        public void InitAdapterMemory() { }

        public void InitWorldInputs(int nGPU) { }

        public void MapWorldInputs() { }

        public void InitWorldOutputs(int nGPU) { }

        public void MapWorldOutputs()
        {
            VisualFov.CopyToMemoryBlock(School.VisualFOV, 0, 0, Math.Min(VisualFov.Count, School.VisualDimensionsFov.Width * School.VisualDimensionsFov.Height));
        }

        public void ClearWorld() { }

        public void SetHint(TSHintAttribute attr, float value) { }
        #endregion
    }
}
