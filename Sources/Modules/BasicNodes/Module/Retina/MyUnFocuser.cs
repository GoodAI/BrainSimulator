using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System.ComponentModel;
using System;
using YAXLib;

namespace GoodAI.Modules.Retina
{
    /// <author>GoodAI</author>
    /// <meta>df/jk-retina</meta>
    ///<status>Working</status>
    ///<summary>Crops and resizes input image according to pupil control input.
    ///Pupil control input must contain position and size of focused area.</summary>
    ///<description>
    ///<ul>
    ///  <li>given a lcoation and image it returns patch there</li>
    ///  <li>given a patch and lcoationj it plots patch there</li>
    ///  <li>supports multiple patches at once</li>
    ///  <li>optional retina like format</li>
    /// </ul>
    /// </description>
    
    public class MyUnfocuser : MyAbstractFocuser
    {
      

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            RetinaPtsDefsMask.Count = (Input != null) ? (Input.Count * 2) : 1; // now retina dimension is the input not OutputWidth :-)
        }



        [MyTaskInfo(Disabled = true, OneShot = true), Description("InitRetina")]
        public class MyUnfocuseInitRetinaTask : MyAbstractInitRetinaTask
        {
            public override void Init(int nGPU)
            {
                Owner.RetinaCircles = Owner.Input.ColumnHint;
            }
        }


    
        /// <summary>
        /// given tha patch and location it fit it inot a image, this is similar to canvas.
        /// </summary>
        public class MyUnfocusTask : MyTask<MyUnfocuser>
        {
            private MyCudaKernel m_kernel;

            private int inputWidth, inputHeight, outputWidth, outputHeight;

            public override void Init(int nGPU)
            {
                inputWidth = Owner.Input.ColumnHint;
                inputHeight = Owner.Input.Count / Owner.Input.ColumnHint;

                outputWidth = Owner.Output.ColumnHint;
                outputHeight = Owner.Output.Count / Owner.Output.ColumnHint;

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearAddSubImageKernel");
                m_kernel.SetupExecution(Owner.Output.Count);
            }

            public override void Execute()
            {
                if (Owner.NumberPupilSamples > 1)
                {
                    MyLog.WARNING.WriteLine("MyFocuser:MyUnfocusTask deoes not support multiple pupils input!");
                }
                else
                {
                    Owner.Output.Fill(0);
                    m_kernel.Run(Owner.Output, Owner.Input, Owner.PupilControl, outputWidth, outputHeight, inputWidth, inputHeight);
                }
            }
        };


        /// <summary>
        /// Given a retina patch and its location, it plots it into a postion.
        /// </summary>
        [MyTaskInfo(Disabled = true)]
        public class MyRetinaUnfocusTask : MyTask<MyUnfocuser>
        {
            private MyCudaKernel m_kernel;

            private int outputWidth, outputHeight;

            public override void Init(int nGPU)
            {
                outputWidth = Owner.Output.ColumnHint;
                outputHeight = Owner.Output.Count / Owner.Output.ColumnHint;

                m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Observers\FocuserInputObserver", "RetinaObserver_UnMaskPatchFl");
                m_kernel.SetupExecution(Owner.Output.Count);
            }

            public override void Execute()
            {
                if (Owner.NumberPupilSamples > 1)
                {
                    MyLog.WARNING.WriteLine("MyFocuser:MyRetinaUnfocusTask deoes not support multiple pupils input! (so far)");
                }
                else
                {
                    m_kernel.Run(Owner.Output, outputWidth, outputHeight, Owner.RetinaPtsDefsMask, Owner.RetinaPtsDefsMask.Count / 2, Owner.Input, Owner.PupilControl);
                }
            }
        };



        public MyUnfocuseInitRetinaTask DoInitRetina { get; private set; }
        public MyUnfocusTask DoInverseTransform { get; private set; }
        public MyRetinaUnfocusTask DoRetinaUnfocus { get; private set; }
    }
}
