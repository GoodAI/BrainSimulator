using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.Modules.GameBoy;
using System;
using GoodAI.Core.Task;
using GoodAI.Core;

namespace GoodAI.School.Worlds
{
    public class PongAdapterWorld : MyCustomPongWorld, IWorldAdapter
    {
        private MyCudaKernel m_kernel;
        private MyCudaKernel m_grayscaleKernel;
        
        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }        

        public void InitAdapterMemory()
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 128;
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

        public void InitWorldInputs(int nGPU)
        {
        }


        public void MapWorldInputs()
        {
            // reward setup??
            // ScoreDeltaOutput.CopyToMemoryBlock(schoolWorld.Reward, 0, 0, 1);

            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            ControlsAdapterTemp.CopyFromMemoryBlock(School.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, School.ActionInput.Count));
        }

        public void InitWorldOutputs(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleKernel");
            m_kernel.SetupExecution(256 * 256);

            m_grayscaleKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\ColorScaleObserverSingle", "DrawGrayscaleKernel");
            m_grayscaleKernel.SetupExecution(256 * 256);

        }

        public void MapWorldOutputs()
        {
            // Rescale data from world to wrapper
            m_kernel.Run(Visual, School.Visual, DISPLAY_WIDTH, DISPLAY_HEIGHT, 256, 256);
            m_grayscaleKernel.Run(School.Visual, School.Visual, 256 * 256);

//            Visual.CopyToMemoryBlock(schoolWorld.Visual, 0, 0, Math.Min(Visual.Count, schoolWorld.VisualSize));

            // Copy of structured data
            Event.CopyToMemoryBlock(School.Data, 0, 0, 1);
            BallPosX.CopyToMemoryBlock(School.Data, 0, 1, 1);
            BallPosY.CopyToMemoryBlock(School.Data, 0, 2, 1);
            PaddlePosX.CopyToMemoryBlock(School.Data, 0, 3, 1);
            PaddlePosY.CopyToMemoryBlock(School.Data, 0, 4, 1);
            BinaryEvent.CopyToMemoryBlock(School.Data, 0, 5, 1);

            //schoolWorld.Visual.Dims = VisualPOW.Dims;
            School.DataLength.Fill(6);
        }

        public void ClearWorld()
        {
            UpdateTask.ResetGame();
        }

        public void SetHint(TSHintAttribute attr, float value)
        {
            // some TSHints related to Tetris?
        }

        /*
        public virtual void CreateTasks()
        {

        }
        */

    }
}
