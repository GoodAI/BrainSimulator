using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.Modules.GameBoy;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Task;
using GoodAI.Core;

namespace GoodAI.School.Worlds
{
    [DisplayName("Pong")]
    public class PongAdapterWorld : MyCustomPongWorld, IWorldAdapter
    {
        public Size Viewport
        {
            get { return new Size(DisplayWidth, DisplayHeight); }
            protected set
            {
                DisplayWidth = value.Width;
                DisplayHeight = value.Height;
            }
        }


        private MyCudaKernel m_kernel;
        private MyCudaKernel m_grayscaleKernel;

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


        public SchoolWorld School { get; set; }
        public MyWorkingNode World { get { return this; } }

        public MyTask GetWorldRenderTask()
        {
            return RenderGameTask;
        }

        public override void UpdateMemoryBlocks()
        {
            if (School != null)
            {
                Viewport = new Size(School.VisualDimensions.Width, School.VisualDimensions.Height);
            }

            base.UpdateMemoryBlocks();
        }

        public void InitAdapterMemory()
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 128;
        }

        public void InitWorldInputs(int nGPU)
        { }

        public void MapWorldInputs()
        {
            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            if (School.ActionInput.Owner is DeviceInput)
            {
                School.ActionInput.SafeCopyToDevice();
                ControlsAdapterTemp.Host[0] = School.ActionInput.Host[65];  // D
                ControlsAdapterTemp.Host[1] = School.ActionInput.Host[87];  // S
                ControlsAdapterTemp.Host[2] = School.ActionInput.Host[68];  // A 
                ControlsAdapterTemp.SafeCopyToDevice();
            }
            else
            {
                ControlsAdapterTemp.CopyFromMemoryBlock(School.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, School.ActionInput.Count));
            }
        }

        public void InitWorldOutputs(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleKernel");
            m_kernel.SetupExecution(Viewport.Width * Viewport.Height);

            m_grayscaleKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\ColorScaleObserverSingle", "DrawGrayscaleKernel");
            m_grayscaleKernel.SetupExecution(Viewport.Width * Viewport.Height);
        }

        public void MapWorldOutputs()
        {
            // Rescale data from world to wrapper
            m_kernel.Run(Visual, School.Visual, Scene.Width, Scene.Height, Viewport.Width, Viewport.Height);
            m_grayscaleKernel.Run(School.Visual, School.Visual, Viewport.Width * Viewport.Height);

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
    }
}
