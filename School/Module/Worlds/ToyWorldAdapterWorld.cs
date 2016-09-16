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
            if (School.ActionInput.Owner is DeviceInput)
            {
                School.ActionInput.SafeCopyToDevice();
                ControlsAdapterTemp.Host[87] = School.ActionInput.Host[87];  // W - Up
                ControlsAdapterTemp.Host[65] = School.ActionInput.Host[65];  // A - Rotate left
                ControlsAdapterTemp.Host[68] = School.ActionInput.Host[68];  // D - Rotate right
                ControlsAdapterTemp.Host[83] = School.ActionInput.Host[83];  // S - Down
                ControlsAdapterTemp.Host[69] = School.ActionInput.Host[69];  // Q - Strafe left
                ControlsAdapterTemp.Host[81] = School.ActionInput.Host[81];  // E - Strafe right

                ControlsAdapterTemp.SafeCopyToDevice();
            }
            else
                ControlsAdapterTemp.CopyFromMemoryBlock(School.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, School.ActionInput.Count));
        }

        public void InitWorldOutputs(int nGPU)
        {
            //throw new NotImplementedException();
        }

        public void MapWorldOutputs()
        {
            VisualFov.CopyToMemoryBlock(School.VisualFOV, 0, 0, Math.Min(VisualFov.Count, School.VisualDimensionsFov.Width * School.VisualDimensionsFov.Height));
        }

        public void ClearWorld()
        {
            //throw new NotImplementedException();
        }

        public void SetHint(TSHintAttribute attr, float value) { }
        #endregion
    }
}
