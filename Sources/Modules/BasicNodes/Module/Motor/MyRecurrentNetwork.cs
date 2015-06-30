using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using BrainSimulator.Memory;
using BrainSimulator.Signals;
using System.Drawing;
using YAXLib;
using ManagedCuda;
using System.Diagnostics;
using BrainSimulator.Transforms;
using BrainSimulator.NeuralNetwork.Layers;

namespace BrainSimulator.Motor
{
    /// <author>Karol Kuna</author>
    /// <status>Working</status>
    /// <summary>Recurrent network trained by Real-Time Recurrent Learning</summary>
    /// <description>Simple recurrent network with fully recurrent hidden layer trained by Real-Time Recurrent Learning (RTRL) algorithm. <br />
    ///              Parameters:
    ///              <ul>
    ///                 <li>INPUT_UNITS: Read-only number of units in input layer</li>
    ///                 <li>HIDDEN_UNITS: Number of units in hidden layer</li>
    ///                 <li>OUTPUT_UNITS: Read-only number of units in output layer</li>
    ///                 <li>UNITS: Read-only total number of units</li>
    ///                 <li>HIDDEN_UNIT_WEIGHTS: Read-only number of connection weights to hidden layer units</li>
    ///                 <li>OUTPUT_UNIT_WEIGHTS: Read-only number of connection weights to output layer units</li>
    ///                 <li>WEIGHTS: Read-only total number of connection weights</li>
    ///              </ul>
    ///              
    ///              I/O:
    ///              <ul>
    ///                 <li>Input: Input vector copied to activation of input layer units and propagated through the network</li>
    ///                 <li>Target: Desired activation of output layer units </li>
    ///                 <li>Output: Activation of output layer units </li>
    ///              </ul>
    ///              
    ///              Signals:
    ///              <ul>
    ///                 <li>Reset: Resets activation of all network units to zero</li>
    ///              </ul>
    /// </description>
    [YAXSerializeAs("RecurrentNetwork")]
    public class MyRecurrentNetwork : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Target { get { return GetInput(1); } }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("Structure")]
        public ActivationFunctionType ACTIVATION_FUNCTION { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int INPUT_UNITS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int HIDDEN_UNITS { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int OUTPUT_UNITS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int UNITS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int HIDDEN_UNIT_WEIGHTS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int OUTPUT_UNIT_WEIGHTS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int WEIGHTS { get; protected set; }

        [MyPersistable]
        public MyMemoryBlock<float> Weights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> WeightDeltas { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> RTRLDerivatives { get; protected set; } //unit activities with respect of input, forward and recurrent weights in time T, T-1
        public MyMemoryBlock<float> PreviousRTRLDerivatives { get; protected set; }

        public MyMemoryBlock<float> Activations { get; protected set; }
        public MyMemoryBlock<float> PreviousActivations { get; protected set; }
        public MyMemoryBlock<float> ActivationDerivatives { get; protected set; }

        public MyMemoryBlock<float> OutputDeltas { get; protected set; }

        //TASKS
        public MyInitNetworkTask InitNetwork { get; protected set; }
        public MyFeedforwardTask Feedforward { get; protected set; }
        public MyRTRLTask RTRL { get; protected set; }

        //SIGNALS
        public MyResetSignal ResetSignal { get; private set; }
        public class MyResetSignal : MySignal { }

        /// <summary>Initializes network with random weights.</summary>
        [Description("Init Network"), MyTaskInfo(OneShot = true)]
        public class MyInitNetworkTask : MyTask<MyRecurrentNetwork>
        {
            private MyCudaKernel m_kernel;
        
            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
                m_kernel.SetupExecution(1);
            }

            public override void Execute()
            {
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Weights.GetDevice(Owner), 0, 0.25f);

                Owner.Activations.Host[0] = 1.0f; //threshold unit is always active
                for (int i = 1; i < Owner.Activations.Count; i++)
                {
                    Owner.Activations.Host[i] = 0.0f;
                }
                Owner.Activations.SafeCopyToDevice();
                Owner.Activations.CopyToMemoryBlock(Owner.PreviousActivations, 0, 0, Owner.Activations.Count);

                Owner.ActivationDerivatives.Fill(0.0f);
                Owner.WeightDeltas.Fill(0.0f);

                Owner.RTRLDerivatives.Fill(0.0f);
                Owner.PreviousRTRLDerivatives.Fill(0.0f);
            }
        }

        /// <summary>Performs forward pass in te network.</summary>
        [Description("Feedforward"), MyTaskInfo(OneShot = false)]
        public class MyFeedforwardTask : MyTask<MyRecurrentNetwork>
        {
            private MyCudaKernel m_kernel;
        
            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\FeedforwardKernel");
                m_kernel.SetupExecution(Owner.HIDDEN_UNITS + Owner.OUTPUT_UNITS);
            }

            public override void Execute()
            {
                if (Owner.ResetSignal.IsIncomingRised())
                {
                    Owner.Activations.Host[0] = 1.0f; //threshold unit is always active
                    for (int i = 1; i < Owner.Activations.Count; i++)
                    {
                        Owner.Activations.Host[i] = 0.0f;
                    }
                    Owner.Activations.SafeCopyToDevice();
                }
                
                m_kernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_kernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_kernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
                m_kernel.SetConstantVariable("D_HIDDEN_UNIT_WEIGHTS", Owner.HIDDEN_UNIT_WEIGHTS);
                m_kernel.SetConstantVariable("D_ACTIVATION_FUNCTION", (int) Owner.ACTIVATION_FUNCTION);

                Owner.Activations.CopyToMemoryBlock(Owner.PreviousActivations, 0, 0, Owner.Activations.Count);
                Owner.Input.CopyToMemoryBlock(Owner.Activations, 0, 1, Owner.Input.Count);

                m_kernel.Run(
                    Owner.Activations,
                    Owner.PreviousActivations,
                    Owner.ActivationDerivatives,
                    Owner.Weights
                    );

                Owner.Activations.CopyToMemoryBlock(Owner.Output, 1 + Owner.INPUT_UNITS + Owner.HIDDEN_UNITS, 0, Owner.OUTPUT_UNITS);
            }
        }

        /// <summary>Computes RTRL partial derivatives and updates network weights.</summary>
        [Description("Real-time recurrent learning"), MyTaskInfo(OneShot = false)]
        public class MyRTRLTask : MyTask<MyRecurrentNetwork>
        {
            private MyCudaKernel m_kernel;
        
            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Structure")]
            public float LEARNING_RATE { get; set; }

            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
            public float MOMENTUM_RATE { get; set; }

            private MyCudaKernel m_changeWeightsKernel;
            private MyCudaKernel m_outputDeltaKernel;

            public override void Init(int nGPU) {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\RTRLDerivativeKernel");
                m_kernel.SetupExecution(Owner.WEIGHTS * (Owner.HIDDEN_UNITS + Owner.OUTPUT_UNITS));

                m_outputDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\OutputDeltaKernel");
                m_outputDeltaKernel.SetupExecution(Owner.OUTPUT_UNITS);

                m_changeWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\ChangeWeightsKernel");
                m_changeWeightsKernel.SetupExecution(Owner.WEIGHTS);
            }

            public override void Execute()
            {
                m_kernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_kernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_kernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
                m_kernel.SetConstantVariable("D_UNITS", Owner.UNITS);
                m_kernel.SetConstantVariable("D_HIDDEN_UNIT_WEIGHTS", Owner.HIDDEN_UNIT_WEIGHTS);
                m_kernel.SetConstantVariable("D_OUTPUT_UNIT_WEIGHTS", Owner.OUTPUT_UNIT_WEIGHTS);
                m_kernel.SetConstantVariable("D_WEIGHTS", Owner.WEIGHTS);

                m_outputDeltaKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_outputDeltaKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_outputDeltaKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

                m_changeWeightsKernel.SetConstantVariable("D_LEARNING_RATE", LEARNING_RATE);
                m_changeWeightsKernel.SetConstantVariable("D_MOMENTUM_RATE", MOMENTUM_RATE);

                m_changeWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_changeWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_changeWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
                m_changeWeightsKernel.SetConstantVariable("D_UNITS", Owner.UNITS);
                m_changeWeightsKernel.SetConstantVariable("D_HIDDEN_UNIT_WEIGHTS", Owner.HIDDEN_UNIT_WEIGHTS);
                m_changeWeightsKernel.SetConstantVariable("D_OUTPUT_UNIT_WEIGHTS", Owner.OUTPUT_UNIT_WEIGHTS);
                m_changeWeightsKernel.SetConstantVariable("D_WEIGHTS", Owner.WEIGHTS);

                Owner.RTRLDerivatives.CopyToMemoryBlock(Owner.PreviousRTRLDerivatives, 0, 0, Owner.RTRLDerivatives.Count);
                
                m_kernel.Run(
                    Owner.Activations,
                    Owner.PreviousActivations,
                    Owner.ActivationDerivatives,
                    Owner.Weights,
                    Owner.RTRLDerivatives,
                    Owner.PreviousRTRLDerivatives
                    );

                m_outputDeltaKernel.Run(Owner.Activations, Owner.Target, Owner.OutputDeltas);
                m_changeWeightsKernel.Run(Owner.RTRLDerivatives, Owner.OutputDeltas, Owner.WeightDeltas, Owner.Weights);
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null && Target != null)
            {
                INPUT_UNITS = Input.Count;
                OUTPUT_UNITS = Target.Count;

                UNITS = 1 + INPUT_UNITS + HIDDEN_UNITS + OUTPUT_UNITS;
                HIDDEN_UNIT_WEIGHTS = HIDDEN_UNITS * (1 + INPUT_UNITS + HIDDEN_UNITS);
                OUTPUT_UNIT_WEIGHTS = OUTPUT_UNITS * (1 + HIDDEN_UNITS);
                WEIGHTS = HIDDEN_UNIT_WEIGHTS + OUTPUT_UNIT_WEIGHTS;

                Output.Count = OUTPUT_UNITS;

                Activations.Count = UNITS; //threshold + all units activation in current time step
                PreviousActivations.Count = Activations.Count;
                ActivationDerivatives.Count = Activations.Count;

                //hidden units are connected to threshold, input and recurrently to themselves; output units are connected to threshold and hidden units
                Weights.Count = WEIGHTS;               
                WeightDeltas.Count = WEIGHTS;

                RTRLDerivatives.Count = WEIGHTS * (HIDDEN_UNITS + OUTPUT_UNITS); //unit activities with respect to weights
                PreviousRTRLDerivatives.Count = RTRLDerivatives.Count;

                OutputDeltas.Count = OUTPUT_UNITS;

                // make an even number of weights for the cuda random initialisation
                if (Weights.Count % 2 != 0)
                    Weights.Count++;
            }
        }
    }
}
