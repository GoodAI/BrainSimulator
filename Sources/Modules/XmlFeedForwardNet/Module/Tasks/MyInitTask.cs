using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Networks;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using BrainSimulator;

namespace  XmlFeedForwardNet.Tasks
{
    /// <summary>
    /// This task initializes the network according to the parameters in the xml definition file.
    /// </summary>
    [Description("Initialization"), MyTaskInfo(Order = 0, OneShot = true)]
    public class MyInitTask : MyTask<MyXMLNetNode>
    {
        public Int32 nGPU;
        public override void Init(Int32 nGPU)
        {
            this.nGPU = nGPU;
            Owner.m_copyKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CopyKernel");
            Owner.m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
        }

        public override void Execute()
        {
            Owner.InputLayer.Initialize(nGPU);
            foreach (MyAbstractFLayer layer in Owner.Layers)
            {
                layer.Initialize(nGPU);
            }
            Owner.DataDimsMemoryBlock.SafeCopyToDevice();
            Owner.ExtraMemoryBlock.SafeCopyToDevice();

            if (Owner.WeightChangesMemoryBlock.Count > 0)
            {
                // Weight change vector
                Owner.WeightChangesMemoryBlock.Fill(0);
                Owner.LastWeightDeltasMemoryBlock.Fill(0);
            }

            Owner.m_currentSamplePosition = 0;
            Owner.SamplesProcessed = 0;

            Owner.RBMBiasMomentum1.Fill(0);
            Owner.RBMBiasMomentum2.Fill(0);
            Owner.RBMWeightMomentum.Fill(0);
        }
    }
}
