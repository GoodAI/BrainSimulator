using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.Common
{
    public enum MyGenerateType
    {
        Linear,
        Sine,
        Cosine,
        UserData,
        SimulationStep,
        SimulationStepFce,
    }

    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>Samples a linear function values to the output array. 
    /// The output is shifted each step by ShiftSpeed parameter.
    /// </summary>
    /// <description></description>
    public class MyGenerateInput : MyWorkingNode
    {
        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [YAXSerializableField(DefaultValue = 1), YAXElementFor("IO")]
        [MyBrowsable, Category("I/O")]
        public int OutputSize
        {
            get { return m_outputSize; }
            set
            {
                m_outputSize = value;
                UpdateOutput();
            }
        }
        private int m_outputSize;

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("I/O")]
        public int ColumnHint { get; set; }

        [YAXSerializableField(DefaultValue = "")]
        [MyBrowsable, Category("User Data Input")]
        public string UserInput
        {
            get { return m_userInput; }
            set
            {
                m_userDataList = (value.Length > 0)
                    ? value.Trim().Split(',', ' ').Select(a => float.Parse(a, CultureInfo.InvariantCulture)).ToList()
                    : null;

                m_userInput = value;
                UpdateOutput();
            }
        }
        private string m_userInput;

        private List<float> m_userDataList;

        [YAXSerializableField(DefaultValue = MyGenerateType.Linear)]
        [MyBrowsable, Category("User Data Input")]
        public MyGenerateType GenerateType { get; set; }

        public override string Description
        {
            get
            {
                if (GenerateType == MyGenerateType.UserData)
                {
                    if (UserInput.Length > 10)
                    {
                        return UserInput.Substring(0, 10) + " ...";
                    }
                    else
                        return UserInput;
                }
                else if (GenerateType == MyGenerateType.SimulationStep)
                {
                    return "SimulStep";
                }
                else if (GenerateType == MyGenerateType.SimulationStepFce)
                {
                    return "SimulStepFce";
                }
                else if (GenerateType == MyGenerateType.Sine)
                {
                    return "Sine";
                }
                else
                {
                    return base.Description;
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            UpdateOutput();
        }

        private void UpdateOutput()
        {
            Output.ColumnHint = ColumnHint;

            Output.Count = GetOutputSize();
        }

        private int GetOutputSize()
        {
            if ((GenerateType != MyGenerateType.Linear) && (GenerateType != MyGenerateType.UserData))
            {
                return 1;
            }
            
            if (GenerateType == MyGenerateType.UserData)
            {
                return (m_userDataList != null) ? m_userDataList.Count : OutputSize;
            }

            return OutputSize;
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(OutputSize > 0, this, "Invalid OutputSize, must be at least 1");
            validator.AssertError(GenerateType == MyGenerateType.Linear || GenerateType == MyGenerateType.SimulationStep || UserInput.Length != 0, this, "You need to enter some values to UserInput field");
        }

        public MyTransferTask GenerateInput { get; private set; }

        /// <summary>Generates input. Possible methods are:<dl>
        /// <dt><b>Linear</b></dt><dd>Set output to numbers spread evenly between <b>MinValue</b> and <b>MaxValue</b> (including) and shifts them each step by <b>ShiftSpeed</b> positions. If the output size is 1, MinValue will be set as output.</dd>
        /// <dt><b>Sine</b></dt><dd>Sets first output element to sine of 2*Pi*SimulationStep*UserInput. Therefore first value in UserInput serves as inverse "sampling frequency". Setting it to 1 or 0.5 will give you just zeros while setting it to 0.01 will spread one sine period to 100 simulation steps.</dd>
        /// <dt><b>Cosine</b></dt><dd>Sets first output element to cosine of 2*Pi*SimulationStep*UserInput. Therefore first value in UserInput serves as inverse "sampling frequency". Setting it to 1 will give you just ones while setting it to 0.01 will spread one cosine period to 100 simulation steps.</dd>
        /// <dt><b>UserData</b></dt><dd>Data in node's <b>UserInput</b> are copied into output</dd>
        /// <dt><b>SimulationStep</b></dt><dd>Output is set to current simulation step number</dd>
        /// <dt><b>SimulationStepFce</b></dt><dd>Task cyclically iterates over node's UserInput and puts values from it to output</dd>
        /// </dl></summary>
        [Description("Generate input")]
        public class MyTransferTask : MyTask<MyGenerateInput>
        {
            private MyCudaKernel m_kernel;

            [YAXSerializableField(DefaultValue = 0)]
            [MyBrowsable, Category("Interval"), DisplayName("M\tinValue")]
            public float MinValue { get; set; }

            [YAXSerializableField(DefaultValue = 1)]
            [MyBrowsable, Category("Interval")]
            public float MaxValue { get; set; }

            [YAXSerializableField(DefaultValue = 0)]
            [MyBrowsable, Category("Interval")]
            public int ShiftSpeed { get; set; }


            public MyTransferTask()
            {
            }

            public override void Init(Int32 nGPU)
            {
                switch (Owner.GenerateType)
                {
                    case MyGenerateType.Linear:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "LinearValuesKernel");
                        break;
                }
            }

            public override void Execute()
            {

                switch (Owner.GenerateType)
                {
                    case MyGenerateType.Linear:
                        m_kernel.SetupExecution(Owner.OutputSize);
                        m_kernel.Run(MinValue, MaxValue, Owner.Output, Owner.OutputSize, ShiftSpeed * SimulationStep);
                        break;
                    case MyGenerateType.Sine:
                        Owner.Output.Host[0] = (float)Math.Sin(this.SimulationStep * 2 * Math.PI * Owner.m_userDataList[0]);
                        Owner.Output.SafeCopyToDevice();
                        break;
                    case MyGenerateType.Cosine:
                        Owner.Output.Host[0] = (float)Math.Cos(this.SimulationStep * 2 * Math.PI * Owner.m_userDataList[0]);
                        Owner.Output.SafeCopyToDevice();
                        break;
                    case MyGenerateType.UserData:
                        for (int a = 0; a < Owner.m_userDataList.Count; a++)
                        {
                            Owner.Output.Host[a] = Owner.m_userDataList[a];
                        }
                        Owner.Output.SafeCopyToDevice();
                        break;
                    case MyGenerateType.SimulationStep:
                        Owner.Output.Host[0] = SimulationStep;
                        Owner.Output.SafeCopyToDevice();
                        break;
                    case MyGenerateType.SimulationStepFce:
                        int stepMod = (int)SimulationStep % Owner.m_userDataList.Count;
                        Owner.Output.Host[0] = Owner.m_userDataList[stepMod];
                        Owner.Output.SafeCopyToDevice();
                        break;
                }
            }

        }

    }
}
