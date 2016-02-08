using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public class DeviceInput : MyWorkingNode
    {
        // The keyboard keys should fit into a short (add some fields for continuous inputs, like mouse).
        // The mapping should match: https://msdn.microsoft.com/en-us/library/windows/desktop/dd375731(v=vs.85).aspx
        private const int TotalOutputSize = 256;

        [MyBrowsable]
        [YAXSerializableField(DefaultValue = false)]
        public bool StepOnKeyDown { get; set; }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = TotalOutputSize;
        }

        public void SetKeyUp(int keyValue)
        {
            Output.Host[keyValue] = 0.0f;

            ExecuteIfPaused();
        }

        public void SetKeyDown(int keyValue)
        {
            bool wasNotPressed = Output.Host[keyValue] < 1.0f;
            Output.Host[keyValue] = 1.0f;

            ExecuteIfPaused();

            MySimulationHandler handler = Owner.SimulationHandler;
            if (wasNotPressed && StepOnKeyDown && (handler.State == MySimulationHandler.SimulationState.PAUSED))
                handler.StartSimulation(1);
        }

        private void ExecuteIfPaused()
        {
            if (Owner.SimulationHandler.State == MySimulationHandler.SimulationState.PAUSED)
                ProcessInputTask.Execute();
        }

        private DeviceInputTask ProcessInputTask { get; set; }

        public class DeviceInputTask : MyTask<DeviceInput> 
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.Output.SafeCopyToDevice();
            }
        }
    }
}
