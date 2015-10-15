using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Modules.Retina;

namespace GoodAI.Modules.Observers
{

    public class MyFocuserRetinaShowPtsMask : MyNodeObserver<MyFocuser>
    {
        MyCudaKernel m_kernel_fillImage;
        public MyFocuserRetinaShowPtsMask()
        {
            m_kernel_fillImage = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\VisionObsFce", "FillVBOFromInputImage");
            m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\FocuserInputObserver", "RetinaObserver_Mask");
        }

        protected override void Execute()
        {
            m_kernel_fillImage.SetupExecution(Target.InputSize);
            m_kernel_fillImage.Run(Target.Input, TextureWidth * TextureHeight, VBODevicePointer);

            m_kernel.SetupExecution(Target.OutputSize);
            m_kernel.Run(VBODevicePointer, TextureWidth, TextureHeight, Target.RetinaPtsDefsMask, Target.OutputSize, Target.PupilControl);
        }

        protected override void Reset()
        {
            TextureWidth = Target.Input.ColumnHint;
            TextureHeight = Target.InputSize / TextureWidth;
        }
    }

    public class MyFocuserRetinaShowPatchInImage : MyNodeObserver<MyFocuser>
    {
        public MyFocuserRetinaShowPatchInImage()
        {
            m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\FocuserInputObserver", "RetinaObserver_UnMaskPatchVBO");
        }        

        protected override void Execute()
        {

            m_kernel.SetupExecution(TextureWidth*TextureHeight);
            m_kernel.Run(VBODevicePointer, TextureWidth, TextureHeight, Target.RetinaPtsDefsMask, Target.RetinaPtsDefsMask.Count/ 2, Target.Output, Target.PupilControl);
        }        

        protected override void Reset()
        {
            TextureWidth = Target.Input.ColumnHint;
            TextureHeight = Target.InputSize / TextureWidth;         
        }
    }
}
