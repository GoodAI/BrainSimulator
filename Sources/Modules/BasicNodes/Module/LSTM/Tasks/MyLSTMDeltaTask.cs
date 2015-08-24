using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using ManagedCuda.BasicTypes;
using YAXLib;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Core;


namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Computes deltas of output gates and cell state errors.</summary>
    [Description("Calculate deltas"), MyTaskInfo(OneShot = false)]
    public class MyLSTMDeltaTask : MyAbstractBackDeltaTask<MyLSTMLayer>
    {
        public MyLSTMDeltaTask() { }

        private MyCudaKernel m_deltaKernel;
        private MyCudaKernel m_deltaBackKernel;

        public override void Init(int nGPU)
        {
            m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaKernel");
            m_deltaBackKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaBackKernel");
            m_deltaKernel.SetupExecution(Owner.MemoryBlocks);
        }

        public override void Execute()
        {
            if (SimulationStep == 0) return;

            // propagate delta to output gates
            m_deltaKernel.Run(
                Owner.Delta,
                Owner.CellStates,
                Owner.PreviousCellStates,
                Owner.CellStateErrors,

                Owner.OutputGateDeltas,
                Owner.ForgetGateDeltas,
                Owner.InputGateDeltas,

                Owner.OutputGateActivations,
                Owner.ForgetGateActivations,
                Owner.InputGateActivations,

                Owner.CellInputActivationDerivatives,
                Owner.OutputGateActivationDerivatives,
                Owner.ForgetGateActivationDerivatives,
                Owner.InputGateActivationDerivatives,

                Owner.CellInputWeights,
                Owner.OutputGateWeights,
                Owner.ForgetGateWeights,
                Owner.InputGateWeights,

                Owner.Input.Count,
                Owner.CellStates.Count,
                Owner.CellsPerBlock );

            // pointer to previous layer
            // MyAbstractLayer previousLayer = Owner.PreviousTopologicalLayer;  // ??? MARTIN HACK TO PREV> LAYER... ADD IT TO FINAL??
            MyAbstractLayer previousLayer = Owner.PreviousLayer;

            if (previousLayer != null)
            {
                // reset delta
                previousLayer.Delta.Fill(0);

                // determine input to previous layer
                CUdeviceptr prevInputPtr;
                if (previousLayer is MyAbstractWeightLayer)
                    prevInputPtr = (previousLayer as MyAbstractWeightLayer).NeuronInput.GetDevicePtr(previousLayer.GPU);
                else
                    prevInputPtr = previousLayer.Input.GetDevicePtr(previousLayer.GPU);

                // propagate delta to previous layer
                m_deltaBackKernel.SetupExecution(previousLayer.Neurons);
                m_deltaBackKernel.Run(
                    (int)previousLayer.ActivationFunction,
                    previousLayer.Delta,
                    Owner.OutputGateDeltas,
                    Owner.ForgetGateDeltas,
                    Owner.InputGateDeltas,

                    Owner.CellInputWeights,
                    Owner.InputGateWeights,
                    Owner.ForgetGateWeights,
                    Owner.OutputGateWeights,

                    previousLayer.Neurons,
                    Owner.CellStates.Count,
                    Owner.CellsPerBlock);
            }
        }

        //public override void Execute()
        //{
        //    if (SimulationStep == 0) return;

        //    // propagate delta to output gates
        //    m_deltaKernel.Run(
        //        Owner.CellStateErrors,
        //        Owner.OutputGateDeltas,
        //        Owner.CellStates,
        //        Owner.OutputGateActivations,
        //        Owner.OutputGateActivationDerivatives,
        //        Owner.Delta,

        //        Owner.CellStates.Count,
        //        Owner.CellsPerBlock
        //        );

        //    // pointer to previous layer
        //    MyAbstractLayer previousLayer = Owner.PreviousTopologicalLayer;

        //    if (previousLayer != null)
        //    {
        //        // reset delta
        //        previousLayer.Delta.Fill(0);

        //        // determine input to previous layer
        //        CUdeviceptr prevInputPtr;
        //        if (previousLayer is MyAbstractWeightLayer)
        //            prevInputPtr = (previousLayer as MyAbstractWeightLayer).NeuronInput.GetDevicePtr(previousLayer.GPU);
        //        else
        //            prevInputPtr = previousLayer.Input.GetDevicePtr(previousLayer.GPU);

        //        // propagate delta to previous layer
        //        m_deltaBackKernel.SetupExecution(previousLayer.Neurons);
        //        m_deltaBackKernel.Run(
        //            (int)previousLayer.ActivationFunction,
        //            prevInputPtr,
        //            previousLayer.Delta,
        //            Owner.CellStateErrors,
        //            Owner.PreviousCellStates,
        //            Owner.InputGateActivations,
        //            Owner.CellInputActivationDerivatives,
        //            Owner.InputGateActivationDerivatives,
        //            Owner.ForgetGateActivationDerivatives,
        //            Owner.CellInputWeights,
        //            Owner.InputGateWeights,
        //            Owner.ForgetGateWeights,
        //            Owner.OutputGateWeights,
        //            Owner.OutputGateDeltas,

        //            previousLayer.Neurons,
        //            Owner.CellStates.Count,
        //            Owner.CellsPerBlock
        //            );
        //    }
        //}
    }
}