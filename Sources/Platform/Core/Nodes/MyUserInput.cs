using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Globalization;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <status>Working</status>
    /// <summary>UserInputNode provides variable number of sliders for manual user input.</summary>
    /// <description>You can set the number of sliders by setting the <b>OutputSize</b> property.
    /// If you set the <b>ConvertToBinary</b> property then only the forst slider is used for the output and one value of the output block will be set to one proportionally to the slider location.
    /// You can control bounds of your inputs with <b>MinValue</b> and <b>MaxValue</b> properties.</description>
    public class MyUserInput : MyWorkingNode
    {
        private bool m_paused; 

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("IO")]
        public int OutputSize
        {
            get { return Output.Count; }
            set { Output.Count = value; }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1)]
        public int ColumnHint { get; set; }

        [MyBrowsable, Category("UI")]
        [YAXSerializableField(DefaultValue = false), YAXElementFor("IO")]
        public bool ShowValues { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("Interval"),DisplayName("M\tinValue")]
        public float MinValue { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("Interval")]
        public float MaxValue { get; set; }

        private float[] m_userInput;

        [YAXSerializableField(DefaultValue = "")]
        private string UserInputStr
        {
            get { return UserInputToString(); }
            set { ParseUserInput(value); }
        }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("Interval")]
        public bool ConvertToBinary { get; set; }

        public override void UpdateMemoryBlocks()
        {
            if (ConvertToBinary)
            {
                if (m_userInput == null || m_userInput.Length != 1)
                {
                    m_userInput = new float[1];
                }
            }
            else
            {
                if (m_userInput == null || m_userInput.Length != OutputSize)
                {
                    m_userInput = new float[OutputSize];
                }
            }

            Output.ColumnHint = ColumnHint;
        }

        public override void Validate(MyValidator validator) 
        {            
            validator.AssertError(MinValue != MaxValue && MinValue < MaxValue, this, "Invalid MinValue and MaxValue combination");
        }

        public override void OnSimulationStateChanged(MySimulationHandler.StateEventArgs args)
        {
            m_paused = args.NewState == MySimulationHandler.SimulationState.PAUSED;
        }

        public void SetUserInput(int index, float value) 
        {
            if (ConvertToBinary)
            {
                m_userInput[0] = value;
            }
            else
            {
                if (m_userInput != null && index < m_userInput.Length)
                {
                    m_userInput[index] = value * (MaxValue - MinValue) + MinValue;
                }
            }

            if (m_paused) 
            {
                if (GenerateInput != null && GenerateInput.Enabled)
                {
                    GenerateInput.Execute();
                }
            }
        }

        public float GetUserInput(int index)
        {
            if (m_userInput != null && index >= 0 && index < m_userInput.Length)
            {
                return m_userInput[index];
            }
            else return 0;
        }

        private string UserInputToString()
        {
            string result = "";

            if (m_userInput != null)
            {
                for (int i = 0; i < m_userInput.Length; i++)
                {                    
                    string numStr = m_userInput[i].ToString(CultureInfo.InvariantCulture);
                    result += i == 0 ? numStr : ";" + numStr; 
                }
            }
            return result;
        }

        private void ParseUserInput(string strValue)
        {
            if (!string.IsNullOrEmpty(strValue)) 
            {
                string[] tokens = strValue.Split(';');

                if (m_userInput == null)
                {
                    m_userInput = new float[tokens.Length];
                }

                int size = Math.Min(tokens.Length, m_userInput.Length);

                for (int i = 0; i < size; i++)
                {
                    m_userInput[i] = float.Parse(tokens[i], CultureInfo.InvariantCulture);
                }
            }            
        }

        public MyTransferTask GenerateInput { get; protected set; }   

        /// <summary>
        /// This task will generate your inputs into the output memory block.
        /// </summary>
        [Description("Generate input")]
        public class MyTransferTask : MyTask<MyUserInput>
        {
            private MyCudaKernel m_kernel;

            public MyTransferTask()
            {
            }

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\IntervalToBinaryVector");
            }

            public override void Execute()
            {
                if (Owner.ConvertToBinary)
                {
                    m_kernel.SetupExecution(Owner.OutputSize);
                    m_kernel.Run(Owner.m_userInput[0], Owner.Output, Owner.OutputSize);
                }
                else
                {
                    Array.Copy(Owner.m_userInput, Owner.Output.Host, Owner.OutputSize);                    
                    Owner.Output.SafeCopyToDevice();
                }                
            }
        }
    }
}
