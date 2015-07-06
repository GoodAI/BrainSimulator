using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Modules.Transforms;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Core;

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

        public override void UpdateMemoryBlocks()
        {
            OutputSize = OutputWidth * OutputWidth;
            Output.ColumnHint = OutputWidth;

            TempPupilControl.Count = 3;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (PupilControl != null)
            {
                validator.AssertError(PupilControl.Count > 2, this, "Not enough control values (at least 3 values needed)");                
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
                outputHeight = Owner.Output.Count / Owner.Output.ColumnHint; 

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleSubImageKernel");
                m_kernel.SetupExecution(Owner.OutputSize);
            }

            public override void Execute()
            {
                Owner.Output.Fill(SafeBounds_FillInValue);
                m_kernel.Run(Owner.Input, Owner.Output, Owner.PupilControl, SafeBounds ? 1 : 0, inputWidth, inputHeight, outputWidth, outputHeight);
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
                Owner.Output.Fill(0);                
                m_kernel.Run(Owner.Output, Owner.Input, Owner.PupilControl, outputWidth, outputHeight, inputWidth, inputHeight);                 
            }
        };

        public MyFocuserTask DoTransform { get; private set; }
        public MyUnfocusTask DoInverseTransform { get; private set; }
    }
}
