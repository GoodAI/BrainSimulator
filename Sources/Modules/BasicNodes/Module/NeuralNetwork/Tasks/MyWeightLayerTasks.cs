using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>
    ///     Initialises the layer parameters with chosen parameters.
    ///     It is recommended to use normal distribution with automatic standard deviation (1.0/sqrt(Input.Count)).
    /// <br></br>
    /// This gives a high certainty, that the neurons don't start out saturated.
    /// </summary>
    /// <description></description>
    [Description("InitWeights"), MyTaskInfo(OneShot = true)]
    public class MyInitWeightsTask : MyTask<MyAbstractWeightLayer>
    {

        private MyCudaKernel m_polynomialKernel;

        public override void Init(int nGPU)
        {
            m_polynomialKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
        }


        public enum RandomDistribution
        {
            Uniform,
            Normal,
            Constant
        }

        //Choose distribution
        [MyBrowsable, Category("\t\tParams")]
        [YAXSerializableField(DefaultValue = RandomDistribution.Normal)]
        public RandomDistribution Distribution { get; set; }


        //Minimal value
        [MyBrowsable, Category("Uniform distribution"), DisplayName("M\tinValue")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float MinValue { get; set; }

        //Maximum value
        [MyBrowsable, Category("Uniform distribution")]
        [YAXSerializableField(DefaultValue = 1f)]
        public float MaxValue { get; set; }

        //Mean for normal dist.
        [MyBrowsable, Category("\tNormal distribution")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float Mean { get; set; }

        //StdDev for normal dist.
        [MyBrowsable, Category("\tNormal distribution")]
        [YAXSerializableField(DefaultValue = 0.01f)]
        public float StdDev { get; set; }

        [MyBrowsable, Category("\tNormal distribution")]
        [YAXSerializableField(DefaultValue = true)]
        [Description("Automatically sets standard deviation to 1.0/sqrt(Input.Count).\nOverrides the StdDev parameter.")]
        public bool AutomaticStdDev { get; set; }

        //Constant value
        [MyBrowsable, Category("Constant distribution")]
        [YAXSerializableField(DefaultValue = 1f)]
        //public float Constant { get; set; }
        public float WeightValue { get; set; }

        [MyBrowsable, Category("Constant distribution")]
        [YAXSerializableField(DefaultValue = 1f)]
        //public float Constant { get; set; }
        public float BiasValue { get; set; }


        public override void Execute() //Task execution
        {
            // init vars to 0
            Owner.PreviousBiasDelta.Fill(0);
            Owner.PreviousWeightDelta.Fill(0);

            // init random weights

            switch (Distribution)
            {
                case RandomDistribution.Uniform:
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.Weights.GetDevice(Owner));
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.Bias.GetDevice(Owner));
                    if (MinValue != 0 && MaxValue != 1)
                    {
                        //scale from 0-1 to min-max
                        m_polynomialKernel.SetupExecution(Owner.Weights.Count);
                        m_polynomialKernel.Run(0, 0, (MaxValue - MinValue), MinValue,
                            Owner.Weights,
                            Owner.Weights,
                            Owner.Weights.Count
                        );
                        //scale from 0-1 to min-max
                        m_polynomialKernel.SetupExecution(Owner.Bias.Count);
                        m_polynomialKernel.Run(0, 0, (MaxValue - MinValue), MinValue,
                            Owner.Bias,
                            Owner.Bias,
                            Owner.Bias.Count
                        );
                    }
                    break;
                case RandomDistribution.Normal:
                    float stdDev = StdDev;
                    if (AutomaticStdDev && Owner.Input != null)
                        stdDev = 1.0f / (float)Math.Sqrt(Owner.Input.Count + 1);
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Weights.GetDevice(Owner), Mean, stdDev);
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Bias.GetDevice(Owner), Mean, stdDev);
                    break;
                case RandomDistribution.Constant:
                    Owner.Weights.Fill(WeightValue);
                    Owner.Bias.Fill(BiasValue);
                    break;
                default:
                    MyLog.WARNING.WriteLine("No initialization distribution set.");
                    return;
            }
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
