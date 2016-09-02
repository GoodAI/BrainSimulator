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

        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }

        #region Implementation of IWorldAdapter
        public MyTask GetWorldRenderTask()
        {
            return UpdateTask;
        }

        public override MyMemoryBlock<float> GetInput(int index)
        {
            return ControlsAdapterTemp;
        }

        public override MyMemoryBlock<T> GetInput<T>(int index)
        {
            return ControlsAdapterTemp as MyMemoryBlock<T>;
        }

        public override MyAbstractMemoryBlock GetAbstractInput(int index)
        {
            return ControlsAdapterTemp;
        }

        public void InitAdapterMemory()
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 128;
        }

        public void InitWorldInputs(int nGPU)
        {
            //throw new NotImplementedException();
        }

        public void MapWorldInputs()
        {
            //throw new NotImplementedException();
        }

        public void InitWorldOutputs(int nGPU)
        {
            //throw new NotImplementedException();
        }

        public void MapWorldOutputs()
        {
            //throw new NotImplementedException();
        }

        public void ClearWorld()
        {
            //throw new NotImplementedException();
        }

        public void SetHint(TSHintAttribute attr, float value) { }
        #endregion
    }
}
