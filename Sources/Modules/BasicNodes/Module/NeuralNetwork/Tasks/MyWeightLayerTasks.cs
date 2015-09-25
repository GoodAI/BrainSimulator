using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System;
using System.ComponentModel;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    /// Initialises the layer parameters randomly with mean 0 and stdDev: 1 / (sqrt(Input.Count + 1))
    /// <br></br>
    /// This gives a high certainty, that the neurons don't start out saturated.
    /// </summary>
    /// <description></description>
    [Description("InitWeights"), MyTaskInfo(OneShot = true)]
    public class MyInitWeightsTask : MyTask<MyAbstractWeightLayer>
    {
        private Random Rand = new Random();

        public MyInitWeightsTask() { } //parameterless constructor
        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute() //Task execution
        {
            // init vars to 0
            Owner.PreviousBiasDelta.Fill(0);
            Owner.PreviousWeightDelta.Fill(0);
            Owner.BiasInput.Fill(1.0f);

            // set standard deviation
            float stdDev = 0.01f;
            if (Owner.Input != null)
                stdDev = 1.0f / (float)Math.Sqrt(Owner.Input.Count + 1);
                

            // init random weights
            for (int w = 0; w < Owner.Weights.Count; w++)
                Owner.Weights.Host[w] = GetRandomGaussian(0.0f, stdDev);
            Owner.Weights.SafeCopyToDevice(); // copy to device

            // init random biases
            for (int b = 0; b < Owner.Bias.Count; b++)
                Owner.Bias.Host[b] = GetRandomGaussian(0.0f, stdDev);
            Owner.Bias.SafeCopyToDevice(); // copy to device
        }

        private float GetRandomGaussian(float mean, float stdDev)
        {
            float u1 = Convert.ToSingle(Rand.NextDouble()); //these are uniform(0,1) random doubles
            float u2 = Convert.ToSingle(Rand.NextDouble()); //these are uniform(0,1) random doubles
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2)); //random normal(0,1)
            return mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
        }
    }

    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    /// Creates a dropout mask for the layer according to the Dropout property of the Neural Network Group
    /// </summary>
    /// <description></description>
    [Description("DropoutMask"), MyTaskInfo(OneShot = false)]
    public class MyCreateDropoutMaskTask : MyTask<MyAbstractWeightLayer>
    {
        public MyCreateDropoutMaskTask() { } //parameterless constructor

        private MyCudaKernel m_dropoutMaskKernel; // kernel
        public override void Init(int nGPU)
        {
            m_dropoutMaskKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "DropoutMaskKernel");
        }

        public override void Execute() //Task execution
        {
            // skip output layer
            if (Owner is MyOutputLayer)
                return;

            // fill with random numbers 0..1
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.DropoutMask.GetDevice(Owner));

            // round to 0 or 1 according to dropout parameter in group
            m_dropoutMaskKernel.SetupExecution(Owner.DropoutMask.Count);
            m_dropoutMaskKernel.Run(
                Owner.DropoutMask,
                Owner.ParentNetwork.Dropout,
                Owner.DropoutMask.Count
                );
        }
    }

    //[Description("GetL1Term"), MyTaskInfo(OneShot = false)]
    //public class MyGetL1TermTask : MyTask<MyAbstractWeightLayer>
    //{
    //    public MyGetL1TermTask() { } //parameterless constructor

    //    private MyCudaKernel m_termKernel; // kernel
    //    public override void Init(int nGPU)
    //    {
    //        m_termKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L1TermKernel");
    //    }

    //    public override void Execute() //Task execution
    //    {
    //        m_termKernel.SetupExecution(m_termKernel.MAX_THREADS);
    //        m_termKernel.DynamicSharedMemory = m_termKernel.BlockDimensions.x * sizeof(float);
    //        m_termKernel.Run(
    //            Owner.Weights,
    //            Owner.L1Term,
    //            Owner.Weights.Count
    //            );
    //    }
    //}

    //[Description("GetL2Term"), MyTaskInfo(OneShot = false)]
    //public class MyGetL2TermTask : MyTask<MyAbstractWeightLayer>
    //{
    //    public MyGetL2TermTask() { } //parameterless constructor

    //    private MyCudaKernel m_termKernel; // kernel
    //    public override void Init(int nGPU)
    //    {
    //        m_termKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L2TermKernel");
    //    }

    //    public override void Execute() //Task execution
    //    {
    //        m_termKernel.SetupExecution(m_termKernel.MAX_THREADS);
    //        m_termKernel.DynamicSharedMemory = m_termKernel.BlockDimensions.x * sizeof(float);
    //        m_termKernel.Run(
    //            Owner.Weights,
    //            Owner.L2Term,
    //            Owner.Weights.Count
    //            );
    //    }
    //}
}
