using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Modules.Retina;

namespace GoodAI.Modules.Observers
{
    public class MyFocuserInputObserver : MyNodeObserver<MyFocuser>
    {
        public MyFocuserInputObserver()
        {
            m_kernel = MyKernelFactory.Instance.Kernel(@"Observers\FocuserInputObserver");         
        }        

        protected override void Execute()
        {
            m_kernel.SetupExecution(Target.InputSize);
            m_kernel.Run(Target.Input, Target.PupilControl, TextureWidth, TextureHeight, VBODevicePointer);
        }        

        protected override void Reset()
        {
            TextureWidth = Target.Input.ColumnHint;
            TextureHeight = Target.InputSize / TextureWidth;         
        }
    }
}
