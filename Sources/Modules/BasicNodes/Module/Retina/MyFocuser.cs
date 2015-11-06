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
    ///  <li>supports multiple patches at once</li>
    ///  <li>optional retina-like format (http://papers.nips.cc/paper/4089-learning-to-combine-foveal-glimpses-with-a-third-order-boltzmann-machine.pdf)</li>
    /// </ul>
    /// </description>
    public class MyFocuser : MyAbstractFocuser
    {


        [MyTaskInfo(Disabled = true, OneShot = true), Description("InitRetina")]
        public class MyFocuserInitRetinaTask : MyAbstractInitRetinaTask
        {
            public override void Init(int nGPU)
            {
                Owner.RetinaCircles = Owner.OutputWidth;
            }

        }
    

        /// <summary>
        /// Given the input and [x,y,scale] (<-1,1>,<-1,1>,<0,1>) the method returns part of the input image that is at that postion.
        /// <br>
        /// If the pupil input is in the form of [N x 3] it returns multiple patches.
        /// </summary>
        [Description("Focus To Area")]
        public class MyFocuserTask : MyTask<MyFocuser>
        {
            private MyCudaKernel m_kernel;
        
            private int inputWidth, inputHeight, outputWidth, outputHeight;

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = true)]
            public bool SafeBounds { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0f)]
            public float SafeBounds_FillInValue { get; set; }

            public override void Init(int nGPU)
            {
                inputWidth = Owner.Input.ColumnHint;
                inputHeight = Owner.Input.Count / Owner.Input.ColumnHint;

                outputWidth = Owner.Output.ColumnHint;
                outputHeight = Owner.Output.Count / Owner.Output.ColumnHint / Owner.NumberPupilSamples;

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleSubImageKernel_ForManyProposals");
                m_kernel.SetupExecution(Owner.OutputSize);
            }

            public override void Execute()
            {
                Owner.Output.Fill(SafeBounds_FillInValue);
                if (Owner.Input != null)
                {
                    m_kernel.Run(Owner.Input, Owner.Output, Owner.PupilControl, SafeBounds ? 1 : 0,
                        Owner.PupilControl.ColumnHint, inputWidth, inputHeight, outputWidth, outputHeight, Owner.NumberPupilSamples, Owner.Output.Count);
                }
                else
                    MyLog.ERROR.WriteLine("Owner.Input is null.");
            }
        }
        


        
        /// <summary>
        /// Same format as nortmal focuser but returns retina-like result (http://papers.nips.cc/paper/4089-learning-to-combine-foveal-glimpses-with-a-third-order-boltzmann-machine.pdf)
        /// There is a mask at has high density closer to the center of the interest. Mask is sparse set it points whcih is the size of the output now. Each element of the output corresponds to the average value of pixels that are closest to the postion of that point.
        /// </summary>
        [MyTaskInfo(Disabled = true), Description("Retina Transform Focuser")]
        public class MyRetinaTransform : MyTask<MyFocuser>
        {

            private MyCudaKernel m_kernel, m_kernelFirstValue;
            private int inputWidth, inputHeight;

            public override void Init(int nGPU)
            {
                m_kernelFirstValue = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "RetinaTransform_HaveAtLeastOneValueThere");
                m_kernelFirstValue.SetupExecution(Owner.OutputSize);

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "RetinaTransform_FillRetinaAtomic");
                m_kernel.SetupExecution(Owner.Input.Count);

                inputWidth = Owner.Input.ColumnHint;
                inputHeight = Owner.Input.Count / Owner.Input.ColumnHint;
            }

            public override void Execute()
            {
                if (Owner.NumberPupilSamples > 1)
                {
                    MyLog.WARNING.WriteLine("MyFocuser:MyRetinaTransform deoes not support multiple pupils input! (so far)");
                }
                else
                {
                    Owner.RetinaTempCumulateSize.Fill(0);
                    Owner.Output.Fill(0);

                    //--- add value for every pixel
                    m_kernelFirstValue.Run(Owner.PupilControl, Owner.Input, inputWidth, inputHeight,
                        Owner.Output, Owner.Output.Count,
                        Owner.RetinaPtsDefsMask, Owner.RetinaPtsDefsMask.Count / 2, Owner.RetinaPtsDefsMask.ColumnHint,
                        Owner.RetinaTempCumulateSize);

                    //--- add all values (some are avoided, so previous kernel fixes that)
                    m_kernel.Run(Owner.PupilControl, Owner.Input, inputWidth, inputHeight,
                        Owner.Output, Owner.Output.Count,
                        Owner.RetinaPtsDefsMask, Owner.RetinaPtsDefsMask.Count / 2, Owner.RetinaPtsDefsMask.ColumnHint,
                        Owner.RetinaTempCumulateSize);

                    //--- average computed values (so far unefficient)
                    Owner.Output.SafeCopyToHost();
                    Owner.RetinaTempCumulateSize.SafeCopyToHost();
                    for (int i = 0; i < Owner.Output.Count; i++)
                    {
                        if (Owner.RetinaTempCumulateSize.Host[i] != 0)
                        {
                            Owner.Output.Host[i] /= Owner.RetinaTempCumulateSize.Host[i];
                        }
                    }
                    Owner.Output.SafeCopyToDevice();
                }
            }
        }

        public MyFocuserInitRetinaTask DoInitRetina { get; private set; }
        public MyFocuserTask DoTransform { get; private set; }
        public MyRetinaTransform DoRetinaTransform { get; private set; }
    }
}
