using GoodAI.Modules.TetrisWorld;
using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using GoodAI.Core.Utils;
using GoodAI.Core;

namespace GoodAI.School.Worlds
{
    public class TetrisAdapterWorld : TetrisWorld, IWorldAdapter
    {
        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }
        private MyCudaKernel m_kernel;

        public void InitAdapterMemory(SchoolWorld schoolWorld)
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 6;
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

        public virtual void InitWorldInputs(int nGPU, SchoolWorld schoolWorld)
        {

        }

        public virtual void MapWorldInputs(SchoolWorld schoolWorld)
        {
            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            ControlsAdapterTemp.CopyFromMemoryBlock(schoolWorld.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, schoolWorld.ActionInput.Count));
        }

        public virtual void InitWorldOutputs(int nGPU, SchoolWorld schoolWorld)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleKernel");
            m_kernel.SetupExecution(256 * 256 * TARGET_VALUES_PER_PIXEL);

            // m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaTextureKernelNearestNeighbor");
            // m_kernel.SetupExecution(VisualWidth * VisualHeight * 3 ); // TARGET_VALUES_PER_PIXEL);
        }

        public virtual void MapWorldOutputs(SchoolWorld schoolWorld)
        {
            // m_kernel.Run(VisualOutput, 256, 256, 128, 128, schoolWorld.Visual, 256, 256, VisualWidth, VisualHeight);
            m_kernel.Run(VisualOutput, schoolWorld.Visual, VisualWidth, VisualHeight, 256, 256);

            //HACK to make it grayscale
            //schoolWorld.Visual.CopyToMemoryBlock(schoolWorld.Visual, 0, 256 * 256, 256 * 256);
            //schoolWorld.Visual.CopyToMemoryBlock(schoolWorld.Visual, 0, 2 * 256 * 256, 256 * 256);

            // Copy data from world to wrapper
            // VisualOutput.CopyToMemoryBlock(schoolWorld.Visual, 0, 0, Math.Min(VisualOutput.Count, schoolWorld.VisualSize));
            if (BrickAreaOutput.Count > 0)
                BrickAreaOutput.CopyToMemoryBlock(schoolWorld.Data, 0, 0, Math.Min(BrickAreaOutput.Count, schoolWorld.DataSize));
            //schoolWorld.Visual.Dims = VisualPOW.Dims;
            schoolWorld.DataLength.Fill(Math.Min(BrickAreaOutput.Count, schoolWorld.DataSize));
            ScoreDeltaOutput.CopyToMemoryBlock(schoolWorld.Reward, 0, 0, 1);
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
