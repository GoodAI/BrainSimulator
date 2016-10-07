using System;
using System.ComponentModel;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using Utils;

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
            ControlsAdapterTemp.Count = 13;
        }

        public void InitWorldInputs(int nGPU)
        {
            //throw new NotImplementedException();
        }

        public void MapWorldInputs()
        {
            ControlsAdapterTemp.Fill(0);
            if (School.ActionInput.Owner is DeviceInput)
            {
                School.ActionInput.SafeCopyToDevice();

                ControlsAdapterTemp.Host[0] = School.ActionInput.Host[ControlMapper.Idx("up")];  // W - Up
                ControlsAdapterTemp.Host[1] = School.ActionInput.Host[ControlMapper.Idx("down")];  // S - Down
                ControlsAdapterTemp.Host[2] = School.ActionInput.Host[ControlMapper.Idx("rot_left")];  // A - Rotate left
                ControlsAdapterTemp.Host[3] = School.ActionInput.Host[ControlMapper.Idx("rot_right")];  // D - Rotate right
                ControlsAdapterTemp.Host[4] = School.ActionInput.Host[ControlMapper.Idx("left")];  // Q - Strafe left
                ControlsAdapterTemp.Host[5] = School.ActionInput.Host[ControlMapper.Idx("right")];  // E - Strafe right

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
