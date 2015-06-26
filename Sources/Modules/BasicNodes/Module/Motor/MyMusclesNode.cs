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
using System.Drawing;
using YAXLib;
using ManagedCuda;
using BrainSimulator.Transforms;

namespace BrainSimulator.Motor
{
    /// <author>Karol Kuna</author>
    /// <status>Working</status>
    /// <summary>Transforms binary alpha neuron activation into final motor output and backwards</summary>
    /// <description></description>
    [YAXSerializeAs("Muscles")]
    public class MyMusclesNode : MyWorkingNode
    {
        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 3), YAXElementFor("Structure")]
        public int MUSCLE_PAIRS { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 128), YAXElementFor("Structure")]
        public int FIBERS_PER_MUSLE { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0.3f), YAXElementFor("Structure")]
        public float ACTIVATION_NEEDED_FOR_FULL_POWER { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int NEURON_COUNT { get; private set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
        public float GRANULARITY { get; set; }

        [MyInputBlock]
        public MyMemoryBlock<float> AlphaNeuronInput { get { return GetInput(0); } }

        [MyInputBlock]
        public MyMemoryBlock<float> MotorInput { get { return GetInput(1); } }

        [MyOutputBlock]
        public MyMemoryBlock<float> MotorOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> AlphaNeuronOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyPersistable]
        public MyMemoryBlock<float> NeuronSize { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<float> NeuronActivation { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<float> NeuronActivationMax { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<float> MotorOutputToFibers { get; private set; }

        public MyMemoryBlock<float> NeuronSpike { get; private set; }

        //TASKS
        public MyInitNeuronsTask InitNeurons { get; private set; }
        public MyAlphaNeuronsTask AlphaNeurons { get; private set; }

        public static void CPUSum(MyMemoryBlock<float> output, MyMemoryBlock<float> input, int size, int outOffset, int inOffset, int stride)
        {
            output.SafeCopyToHost();
            input.SafeCopyToHost();

            output.Host[outOffset] = 0.0f;

            for (int i = 0; i < size; i++)
            {
                output.Host[outOffset] += input.Host[inOffset + i];
            }

            output.SafeCopyToDevice();
        }

        [Description("Init Neurons"), MyTaskInfo(OneShot = true)]
        public class MyInitNeuronsTask : MyTask<MyMusclesNode>
        {
            private MyCudaKernel m_kernel;
        
            public override void Init(int nGPU)
            {
                m_kernel = MyReductionFactory.Kernel(nGPU, MyReductionFactory.Mode.f_Sum_f);
            }

            public override void Execute()
            {
                //Generate random neuron sizes (bigger neuron contributes more to overall muscle force)
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.NeuronSize.GetDevice(Owner), 1.0f, 0.5f);
                Owner.NeuronSize.SafeCopyToHost();

                //Calculate sum of neurons sizes per each muscle. This is the maximum muscle activation possible. However, this number can be lowered by ACTIVATION_NEEDED_FOR_FULL_POWER
                m_kernel.SetupExecution(Owner.FIBERS_PER_MUSLE / 2);
                for (int i = 0; i < Owner.MUSCLE_PAIRS*2; i++)
                {
                    //m_kernel.Run(
                    BrainSimulator.Motor.MyMusclesNode.CPUSum(
                        Owner.NeuronActivationMax, //output
                        Owner.NeuronSize, //input
                        Owner.FIBERS_PER_MUSLE, //size,
                        i, //outOffset
                        i * Owner.FIBERS_PER_MUSLE, //inOffset
                        1 //stride
                    );
                }
                Owner.NeuronActivationMax.SafeCopyToHost();

                //If motor input is provided, AlphaNeuronCopy output block is filled with alpha neuron activation that could cause it.
                //Allowed motor interval (-1,1) is divided into parts of size GRANULARITY.
                //Since there are many solutions, one is pre-generated for every sub-interval at the beginning and store it in MotorOutputToFibers block.
                Random rand = new Random();
                int subIntervals = ((int)(2 * (1.0f / Owner.GRANULARITY)));

                for (int i = 0; i < Owner.MUSCLE_PAIRS; i++)
                {
                    for (int j = 0; j < subIntervals; j++)
                    {
                        int offset = i*2*Owner.FIBERS_PER_MUSLE*subIntervals + j*2*Owner.FIBERS_PER_MUSLE;
                        //fill array with 0s, only 1s are written to it later
                        for (int k = 0; k < 2*Owner.FIBERS_PER_MUSLE; k++)
                        {
                            Owner.MotorOutputToFibers.Host[offset + k] = 0.0f;
                        }
                        float expectedSum = (j - subIntervals / 2) * Owner.GRANULARITY; //number in (-1,1) interval to be reached
                        if (expectedSum >= 0) expectedSum += Owner.GRANULARITY; //there is no reason to store activations that cause 0 force
                        float realSum = 0.0f;

                        if (j < subIntervals / 2) //negative interval (-1,-Owner.GRANULARITY)
                        {
                            //generate new active neuron id until total activation matches expected activation
                            while (realSum / Owner.NeuronActivationMax.Host[2 * i + 1] > expectedSum * Owner.ACTIVATION_NEEDED_FOR_FULL_POWER)
                            {
                                int r = rand.Next(0, Owner.FIBERS_PER_MUSLE);
                                if (Owner.MotorOutputToFibers.Host[offset + Owner.FIBERS_PER_MUSLE + r] == 0)
                                {
                                    Owner.MotorOutputToFibers.Host[offset + Owner.FIBERS_PER_MUSLE + r] = 1;
                                    realSum -= Owner.NeuronSize.Host[2 * i * Owner.FIBERS_PER_MUSLE + Owner.FIBERS_PER_MUSLE + r];
                                }
                            }
                        }
                        else //positive interval (Owner.GRANULARITY, 1)
                        {
                            //generate new active neuron id until total activation matches expected activation
                            while (realSum / Owner.NeuronActivationMax.Host[2 * i] < expectedSum*Owner.ACTIVATION_NEEDED_FOR_FULL_POWER)
                            {
                                int r = rand.Next(0, Owner.FIBERS_PER_MUSLE);
                                if (Owner.MotorOutputToFibers.Host[offset + r] == 0)
                                {
                                    Owner.MotorOutputToFibers.Host[offset + r] = 1;
                                    realSum += Owner.NeuronSize.Host[2 * i * Owner.FIBERS_PER_MUSLE + r];
                                }
                            }
                        }
                    }
                }

                Owner.MotorOutputToFibers.SafeCopyToDevice();
            }
        }


        [Description("Alpha neurons")]
        public class MyAlphaNeuronsTask : MyTask<MyMusclesNode>
        {
            private MyCudaKernel m_kernel;
            private MyCudaKernel m_multiplyKernel;

            public override void Init(int nGPU) {
                m_kernel = MyReductionFactory.Kernel(nGPU, MyReductionFactory.Mode.f_Sum_f);
                m_multiplyKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");

                m_kernel.SetupExecution(Owner.FIBERS_PER_MUSLE / 2);
                m_multiplyKernel.SetupExecution(Owner.NEURON_COUNT);
            }

            public override void Execute()
            {
                //Motor command -> Alpha neuron activation
                if (Owner.MotorInput != null) //MotorInput is connected, size is already validated
                {
                    Owner.MotorInput.SafeCopyToHost();
                    Owner.AlphaNeuronOutput.Fill(0.0f);

                    //copy pre-computed alpha neuron activation that matches current input to AlphaNeuronCopyOutput
                    int subIntervals = ((int)(2 * (1.0f / Owner.GRANULARITY)));

                    for (int i = 0; i < Owner.MUSCLE_PAIRS; i++)
                    {
                        float value = Owner.MotorInput.Host[i];
                        if (value < -Owner.GRANULARITY)
                        {
                            int index = subIntervals / 2 + (int)(value / Owner.GRANULARITY);
                            int offset = i * 2 * Owner.FIBERS_PER_MUSLE * subIntervals + index * 2 * Owner.FIBERS_PER_MUSLE + Owner.FIBERS_PER_MUSLE;
                            Owner.MotorOutputToFibers.CopyToMemoryBlock(Owner.AlphaNeuronOutput, offset, (2 * i + 1) * Owner.FIBERS_PER_MUSLE, Owner.FIBERS_PER_MUSLE);
                        }
                        else if (value > Owner.GRANULARITY)
                        {
                            int index = subIntervals / 2 - 1 + (int)(value / Owner.GRANULARITY); // - 1 because 0 value is skipped (no activation)
                            int offset = i * 2 * Owner.FIBERS_PER_MUSLE * subIntervals + index * 2 * Owner.FIBERS_PER_MUSLE;
                            Owner.MotorOutputToFibers.CopyToMemoryBlock(Owner.AlphaNeuronOutput, offset, 2 * i * Owner.FIBERS_PER_MUSLE, Owner.FIBERS_PER_MUSLE);
                        }
                    }
                }

                //Alpha neuron activation -> Motor command
                if (Owner.AlphaNeuronInput != null) //AlphaNeuronInput is connected, size is already validated
                {
                    //Owner.NeuronSpike = Owner.AlphaNeuronInput * Owner.NeuronSize
                    m_multiplyKernel.Run(Owner.AlphaNeuronInput,
                        Owner.NeuronSize,
                        Owner.NeuronSpike,
                        2, //Multiply method
                        Owner.NEURON_COUNT
                    );

                    //Owner.NeuronActivation = sum(Owner.NeuronSpike) for every muscle pair
                    for (int i = 0; i < Owner.MUSCLE_PAIRS * 2; i++)
                    {
                        //m_kernel.Run(
                        BrainSimulator.Motor.MyMusclesNode.CPUSum(
                            Owner.NeuronActivation, //output
                            Owner.NeuronSpike, //input
                            Owner.FIBERS_PER_MUSLE, //size,
                            i, //outOffset
                            i * Owner.FIBERS_PER_MUSLE, //inOffset
                            1 //stride
                        );
                    }
                    Owner.NeuronActivation.SafeCopyToHost();

                    //calculate final motor commands
                    for (int i = 0; i < Owner.MUSCLE_PAIRS * 2; i += 2)
                    {
                        Owner.MotorOutput.Host[i / 2] = Owner.NeuronActivation.Host[i] / Owner.NeuronActivationMax.Host[i] - Owner.NeuronActivation.Host[i + 1] / Owner.NeuronActivationMax.Host[i + 1];
                        Owner.MotorOutput.Host[i / 2] = Math.Max(-1.0f, Math.Min(1.0f, Owner.MotorOutput.Host[i / 2] / Owner.ACTIVATION_NEEDED_FOR_FULL_POWER));
                    }
                    Owner.MotorOutput.SafeCopyToDevice();
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            NEURON_COUNT = 2 * MUSCLE_PAIRS * FIBERS_PER_MUSLE;
            NeuronSize.Count = NEURON_COUNT;
            NeuronSpike.Count = NEURON_COUNT;
            NeuronActivation.Count = 2 * MUSCLE_PAIRS;
            NeuronActivationMax.Count = 2 * MUSCLE_PAIRS;

            AlphaNeuronOutput.Count = NEURON_COUNT;
            AlphaNeuronOutput.ColumnHint = FIBERS_PER_MUSLE;

            MotorOutput.Count = MUSCLE_PAIRS;
            MotorOutputToFibers.Count = ((int)(2 * (1.0f / GRANULARITY))) * NEURON_COUNT;
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(AlphaNeuronInput == null || AlphaNeuronInput.Count == NEURON_COUNT, this, "Alpha neuron input size must match number of neurons");
            validator.AssertError(AlphaNeuronInput == null || AlphaNeuronInput.ColumnHint == FIBERS_PER_MUSLE, this, "Alpha neuron input column hint must match number of fibers per muscle");
            validator.AssertError(MotorInput == null || MotorInput.Count == MUSCLE_PAIRS, this, "Alpha neuron input size must match number of neurons");
        }
    }
}
