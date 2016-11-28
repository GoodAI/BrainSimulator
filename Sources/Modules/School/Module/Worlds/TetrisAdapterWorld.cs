using GoodAI.Modules.TetrisWorld;
using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using GoodAI.Core.Utils;
using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.ToyWorld;

namespace GoodAI.School.Worlds
{
    [DisplayName("Tetris")]
    public class TetrisAdapterWorld : TetrisWorld, IWorldAdapter
    {
        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }


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

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(ActionInput != null, this, "ActionInput must not be null");
            if (ActionInput != null)
                validator.AssertError(ActionInput.Count == 6, this, "Size of ActionInput must be 6");
            //base.Validate(validator);
        }


        public SchoolWorld School { get; set; }
        public MyWorkingNode World { get { return this; } }
        public bool CopyDataThroughCPU { get; set; }

        public MyTask GetWorldRenderTask()
        {
            return RenderGameTask;
        }

        public override void UpdateMemoryBlocks()
        {
            if (School != null)
            {
                VisualWidth = School.VisualDimensionsFov.Width;
                VisualHeight = School.VisualDimensionsFov.Height;
				CopyDataThroughCPU = School.CopyDataThroughCPU;
            }

            base.UpdateMemoryBlocks();
        }

        public void InitAdapterMemory()
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 6;
        }

        public virtual void InitWorldInputs(int nGPU)
        { }

        public virtual void MapWorldInputs()
        {
            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            School.ActionInput.SafeCopyToHost();
            ControlsAdapterTemp.Host[1] = School.ActionInput.Host[ControlMapper.Idx("left")]; // A
            ControlsAdapterTemp.Host[2] = School.ActionInput.Host[ControlMapper.Idx("right")]; // D
            ControlsAdapterTemp.Host[3] = School.ActionInput.Host[ControlMapper.Idx("backward")]; // S
            ControlsAdapterTemp.Host[4] = School.ActionInput.Host[ControlMapper.Idx("rot_left")]; // Q
            ControlsAdapterTemp.Host[5] = School.ActionInput.Host[ControlMapper.Idx("rot_right")]; // E
            ControlsAdapterTemp.SafeCopyToDevice();
        }

        public virtual void InitWorldOutputs(int nGPU)
        {
        }

        public virtual void MapWorldOutputs()
        {
            // Copy data from world to wrapper
            VisualOutput.CopyToMemoryBlock(School.VisualFOV, 0, 0, Math.Min(VisualOutput.Count, School.VisualDimensionsFov.Width * School.VisualDimensionsFov.Height));

            if (BrickAreaOutput.Count > 0)
                BrickAreaOutput.CopyToMemoryBlock(School.Data, 0, 0, Math.Min(BrickAreaOutput.Count, School.DataSize));
            School.DataLength.Fill(Math.Min(BrickAreaOutput.Count, School.DataSize));

            ScoreDeltaOutput.CopyToMemoryBlock(School.RewardMB, 0, 0, 1);
        }

        public void ClearWorld()
        {
            Engine.Reset();
        }

        public void SetHint(TSHintAttribute attr, float value)
        {
            // some TSHints related to Tetris?
        }
    }
}
