using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;

namespace GoodAI.Core.Nodes
{
    public class DeviceInput : MyWorkingNode
    {
        // The keyboard keys should fit into a short (add some fields for continuous inputs, like mouse).
        private const int TotalOutputSize = 256;// + 32;

        private float[] m_values = new float[TotalOutputSize];

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

        public void OnKeyUp(int keyValue)
        {
            //MyLog.INFO.WriteLine("KeyUp: {0}", keyValue);
            m_values[keyValue] = 0.0f;
        }

        public void OnKeyDown(int keyValue)
        {
            MyLog.INFO.WriteLine("KeyDown: {0}", keyValue);
            m_values[keyValue] = 1.0f;
        }

        private DeviceInputTask ProcessInputTask { get; set; }

        public class DeviceInputTask : MyTask<DeviceInput> 
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                MyLog.DEBUG.WriteLine("InputDevice - task executing");
                Array.Copy(Owner.m_values, Owner.Output.Host, Owner.m_values.Length);
                Owner.Output.SafeCopyToDevice();
            }
        }
    }
}
