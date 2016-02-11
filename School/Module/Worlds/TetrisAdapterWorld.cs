using GoodAI.Modules.TetrisWorld;
using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using GoodAI.Core.Utils;
using GoodAI.Core;
using GoodAI.Core.Task;

namespace GoodAI.School.Worlds
{
    public class TetrisAdapterWorld : TetrisWorld, IWorldAdapter
    {
        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }

        public void InitAdapterMemory()
        {
            //VisualHeight = VisualWidth = 256;
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 6;
            VisualHeight = VisualWidth = 400;
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

        public MyWorkingNode World
        {
            get { return this; }
        }

        public SchoolWorld School { get; set; }

        public MyTask GetWorldRenderTask()
        { 
            return RenderGameTask; 
        }

        public override void UpdateMemoryBlocks()
        {
            VisualWidth = 256;
            base.UpdateMemoryBlocks();
        }

        public virtual void InitWorldInputs(int nGPU)
        {

        }

        public virtual void MapWorldInputs()
        {
            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            ControlsAdapterTemp.CopyFromMemoryBlock(School.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, School.ActionInput.Count));
        }

        public virtual void InitWorldOutputs(int nGPU)
        {
        }

        public virtual void MapWorldOutputs()
        {
            // Copy data from world to wrapper
            VisualOutput.CopyToMemoryBlock(School.Visual, 0, 0, Math.Min(VisualOutput.Count, School.VisualSize));

            if (BrickAreaOutput.Count > 0)
                BrickAreaOutput.CopyToMemoryBlock(School.Data, 0, 0, Math.Min(BrickAreaOutput.Count, School.DataSize));
            School.DataLength.Fill(Math.Min(BrickAreaOutput.Count, School.DataSize));

            ScoreDeltaOutput.CopyToMemoryBlock(School.Reward, 0, 0, 1);
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
