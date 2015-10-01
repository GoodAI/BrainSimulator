using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Retina
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    ///<status>Working</status>
    ///<summary>Crops and resizes input image according to pupil control input.
    ///Pupil control input must contain position and size of focused area.</summary>
    ///<description></description>
    public class MyFocuser : MyTransform
    {
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 64)]
        public int OutputWidth { get; set; }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> PupilControl 
        {
            get { return GetInput(1); }
        }

        public MyMemoryBlock<float> TempPupilControl { get; private set; }

        public int NumberPupilSamples;

        public override void UpdateMemoryBlocks()
        {
            OutputSize = OutputWidth * OutputWidth;
            Output.ColumnHint = OutputWidth;

            NumberPupilSamples = 1;
            if (PupilControl != null && PupilControl.Count > 3) // for multi input -> set how $ pupils samples from the count
                NumberPupilSamples = PupilControl.Count / PupilControl.ColumnHint;
            OutputSize *= NumberPupilSamples;

            TempPupilControl.Count = 3;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (PupilControl != null)
            {
                validator.AssertError(PupilControl.Count > 2, this, "Not enough control values (at least 3 values needed)");

                validator.AssertError((PupilControl.Count % 3) == 0, this, "Wrong pupil control input size, it has to be [x,y,s] or [x,y,s;x,y,s...]");
                validator.AssertError((float)PupilControl.Count / (float)PupilControl.ColumnHint != 0, this, "If input is matrix, it has to be 3 columns and N rows, each row x,y,s");
            }
        }

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

        public class MyUnfocusTask : MyTask<MyFocuser>
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

        public MyFocuserTask DoTransform { get; private set; }
        public MyUnfocusTask DoInverseTransform { get; private set; }
    }
}
