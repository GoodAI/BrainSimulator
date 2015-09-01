using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;


namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>PID Controller</summary>
    /// <description>Proportional-Integral-Derivative (PID) controller.
    /// Minimises error between measured process variable (Input) and its setpoint (Goal) by adjusting manipulated variable (Output).
    /// PID controls single process variable by single manipulated variable. When trying to control more variables, PID controllers for each variable are independent.<br />
    ///     I/O:
    ///         <ul>
    ///             <li>Input: Measured process variable</li>
    ///             <li>Goal: Desired setpoint of process variable</li>
    ///             <li>Output: Controller output of manipulated variable</li>
    ///         </ul>
    /// 
    /// </description>
    [YAXSerializeAs("Controller")]
    public class MyPIDController : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Goal { get { return GetInput(1); } }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<float> PreviousError { get; private set; }
        public MyMemoryBlock<float> Integral { get; private set; }

        public MyInitTask InitTask { get; protected set; }
        public MyControlTask ControlTask { get; protected set; }

        /// <summary>Initialises node.</summary>
        [Description("Init"), MyTaskInfo(OneShot = true, Order = 0)]
        public class MyInitTask : MyTask<MyPIDController>
        {
            public override void Init(int nGPU) {}

            public override void Execute()
            {
                Owner.PreviousError.Fill(0);
                Owner.Integral.Fill(0);

                Owner.PreviousError.SafeCopyToHost();
                Owner.Integral.SafeCopyToHost();
            }
        }
        /// <summary>PID control to minimise error (Input - Goal) calculted as weighted sum of proportional, integral, and derivative terms.<br />
        ///          Parameters:
        ///              <ul>
        ///                 <li>PROPORTIONAL_GAIN: Weight of gain proportional to current error</li>
        ///                 <li>INTEGRAL_GAIN: Weight of gain from sum of all past errors</li>
        ///                 <li>DERIVATIVE_GAIN: Weight of gain from derivative of current error</li>
        ///                 <li>INTEGRAL_DECAY: Error integral multiplier (Integral[t] = CurrentError[t] + INTEGRAL_DECAY * Integral[t - 1])</li>
        ///                 <li>OFFSET: Offset to be added to controller output</li>
        ///                 <li>MIN_OUTPUT: Lower bound of controller output</li>
        ///                 <li>MAX_OUTPUT: Upper bound of controller output</li>
        ///             </ul>
        /// </summary>
        [Description("Control"), MyTaskInfo(OneShot = false, Order = 0)]
        public class MyControlTask : MyTask<MyPIDController>
        {
            private MyCudaKernel m_kernel;
        
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float PROPORTIONAL_GAIN { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.01f)]
            public float INTEGRAL_GAIN { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float DERIVATIVE_GAIN { get; set; }


            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float INTEGRAL_DECAY { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.0f)]
            public float OFFSET { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = -1.0f)]
            public float MIN_OUTPUT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float MAX_OUTPUT { get; set; }

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\PIDControllerKernel");
                m_kernel.SetupExecution(Owner.Input.Count);
                m_kernel.SetConstantVariable("D_COUNT", Owner.Input.Count);
            }

            public override void Execute()
            {
                m_kernel.SetConstantVariable("D_PROPORTIONAL_GAIN", PROPORTIONAL_GAIN);
                m_kernel.SetConstantVariable("D_INTEGRAL_GAIN", INTEGRAL_GAIN);
                m_kernel.SetConstantVariable("D_DERIVATIVE_GAIN", DERIVATIVE_GAIN);
                m_kernel.SetConstantVariable("D_INTEGRAL_DECAY", INTEGRAL_GAIN);
                m_kernel.SetConstantVariable("D_OFFSET", OFFSET);
                m_kernel.SetConstantVariable("D_MIN_OUTPUT", MIN_OUTPUT);
                m_kernel.SetConstantVariable("D_MAX_OUTPUT", MAX_OUTPUT);

                m_kernel.Run(Owner.Input, Owner.Goal, Owner.Output, Owner.PreviousError, Owner.Integral);
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                Output.Count = Input.Count;
                PreviousError.Count = Input.Count;
                Integral.Count = Input.Count;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Input != null && Goal != null && Input.Count == Goal.Count, this, "Input size must equal goal size!");
        }
    }
}
