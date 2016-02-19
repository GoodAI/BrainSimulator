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

        public MyWorkingNode World { get { return this; } }
        public SchoolWorld School { get; set; }

        public MyTask GetWorldRenderTask()
        {
            return RenderGameTask;
        }

        public override void UpdateMemoryBlocks()
        {
            if (School != null)
            {
                VisualWidth = School.Visual.Dims[0];
                VisualHeight = School.Visual.Dims[1];
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
            if (School.ActionInput.Owner is DeviceInput)
            {
                School.ActionInput.SafeCopyToDevice();
                ControlsAdapterTemp.Host[1] = School.ActionInput.Host[65];  // A - Left
                ControlsAdapterTemp.Host[2] = School.ActionInput.Host[68];  // D - Right
                ControlsAdapterTemp.Host[3] = School.ActionInput.Host[83];  // S - Down
                ControlsAdapterTemp.Host[4] = School.ActionInput.Host[75];  // K - rotate left
                ControlsAdapterTemp.Host[5] = School.ActionInput.Host[76];  // L - rotate right

                ControlsAdapterTemp.SafeCopyToDevice();
            }
            else
            {
                ControlsAdapterTemp.CopyFromMemoryBlock(School.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, School.ActionInput.Count));
            }
        }

        public virtual void InitWorldOutputs(int nGPU)
        {
        }

        public virtual void MapWorldOutputs()
        {
            // Copy data from world to wrapper
            VisualOutput.CopyToMemoryBlock(School.Visual, 0, 0, Math.Min(VisualOutput.Count, School.Visual.Count));

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


        public override void Validate(MyValidator validator)
        {
            validator.AssertError(ActionInput != null, this, "ActionInput must not be null");
            if (ActionInput != null)
                validator.AssertError(ActionInput.Count == 6, this, "Size of ActionInput must be 6");
            //base.Validate(validator);
        }


    }
}
