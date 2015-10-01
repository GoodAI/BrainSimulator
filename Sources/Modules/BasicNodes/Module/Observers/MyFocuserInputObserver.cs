using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Modules.Retina;

namespace GoodAI.Modules.Observers
{
    public class MyFocuserInputObserver : MyNodeObserver<MyFocuser>
    {
        MyCudaKernel m_kernel_fillImage;
        public MyFocuserInputObserver()
        {
            m_kernel_fillImage = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\VisionObsFce", "FillVBOFromInputImage");
            m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\KMeansWM", "FocuserInputObserver");
        }        

        protected override void Execute()
        {
            m_kernel_fillImage.SetupExecution(Target.InputSize);
            m_kernel_fillImage.Run(Target.Input, TextureWidth * TextureHeight, VBODevicePointer);

            m_kernel.SetupExecution(Target.InputSize);
            for (int pupCnt = 0; pupCnt < Target.NumberPupilSamples; pupCnt++)
                m_kernel.Run(Target.Input,Target.PupilControl.GetDevicePtr(Target, pupCnt * Target.PupilControl.ColumnHint), TextureWidth, TextureHeight, VBODevicePointer, ((float)pupCnt)/(float)Target.NumberPupilSamples,1);
         
        }        

        protected override void Reset()
        {
            TextureWidth = Target.Input.ColumnHint;
            TextureHeight = Target.InputSize / TextureWidth;         
        }
    }
}
