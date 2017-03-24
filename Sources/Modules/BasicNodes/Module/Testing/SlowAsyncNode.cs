using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Profiling;
using YAXLib;

namespace GoodAI.Modules.Testing
{
    /// <author>GoodAI</author>
    /// <meta>premek</meta>
    /// <status>Testing</status>
    /// <summary>Performs async CUDA computation.</summary>
    /// <description>
    /// Node intended for testing CUDA multi-streaming.
    /// </description>
    public sealed class SlowAsyncNode : MyWorkingNode
    {

        [MyBrowsable, Category("Parallelization Hacks"), YAXSerializableField(DefaultValue = CudaSyncHelper.NoLayer)]
        [Description("Layer number to provide info for cuda stream synchronization")]
        public int LayerNumber { get; set; } = CudaSyncHelper.NoLayer;

        [MyInputBlock]
        public MyMemoryBlock<float> Input => GetInput(0);

        [MyNonpersistableOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value);}
        }

        public SlowTask Task { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Dims = Input?.Dims ?? TensorDimensions.Empty;
        }
    }

    public class SlowTask : MyTask<SlowAsyncNode>
    {
        private MyCudaKernel m_chewDataKernel;

        private readonly LoggingStopwatch m_stopwatch = new LoggingStopwatch(iterationCountPerBatch: 100);

        public override void Init(int nGPU)
        {
            m_chewDataKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Testing\Stress", "ChewDataKernel");
        }

        public override void Execute()
        {
            CudaSyncHelper.Instance.OnStartExecute(Owner.LayerNumber);

            const int cycleCountThousands = 100;

            m_stopwatch.StartNewSegment($"Chewing {cycleCountThousands} thousand cycles.");

            m_chewDataKernel.SetupExecution(Owner.Input.Count);
            m_chewDataKernel.Run(0.25f, Owner.Input, Owner.Output, Owner.Input.Count, cycleCountThousands);

            m_stopwatch.StopAndSometimesPrintStats();
        }
    }
}
