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

        private static Random Rand = new Random();

        public enum RandomDistribution
        {
            Ones,
            Default
        }

        //Choose distribution
        [MyBrowsable, Category("\t\tParams")]
        [YAXSerializableField(DefaultValue = RandomDistribution.Default)]
        public RandomDistribution Distribution { get; set; }
        
        [MyBrowsable, Category("\t\tParams")]
        [YAXSerializableField(DefaultValue = 1)]
        public float Multiplier { get; set; }

        [MyBrowsable, Category("\t\tParams")]
        [YAXSerializableField(DefaultValue = 1)]
        public float MultiplierBias { get; set; }

        public override void Execute() //Task execution
        {
            // init vars to 0
            Owner.PreviousBiasDelta.Fill(0);
            Owner.PreviousWeightDelta.Fill(0);
            Owner.BiasInput.Fill(1.0f);

            // init weights
            switch (Distribution)
            {
                case RandomDistribution.Ones:
                    Owner.Weights.Fill(1.0f);
                    Owner.Bias.Fill(1.0f);
                    break;

                case RandomDistribution.Default:
                    float stdDev = 1.0f;
                    float Mean = 0.0f;
                    if (Owner.Input != null)
                        stdDev = 1.0f / (float)Math.Sqrt(Owner.Input.Count / Owner.ParentNetwork.BatchSize + 1);

                    // init random weights
                    for (int w = 0; w < Owner.Weights.Count; w++)
                    {
                        Owner.Weights.Host[w] = Multiplier * GetRandomGaussian(0.0f, stdDev);
                    }
                    Owner.Weights.SafeCopyToDevice(); // copy to device

                    // init random biases
                    for (int b = 0; b < Owner.Bias.Count; b++)
                    {
                        Owner.Bias.Host[b] = MultiplierBias * GetRandomGaussian(0.0f, stdDev);
                    }
                    Owner.Bias.SafeCopyToDevice(); // copy to device
                    break;

                default:
                    MyLog.WARNING.WriteLine("No initialization distribution set.");
                    return;
            }
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
    [Description("DropoutMask"), MyTaskInfo(OneShot = false, Disabled = true)]
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

    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>
    /// First it copies the weights from source layer and then slowly updates them to track source weights according to: <br/>
    /// new weights = ApproachRate * sourceLayerWeights + (1 - ApproachRate) * weights <br/>
    /// Set ApproachRate to 1 to use the exact same weights as source layer.
    /// </summary>
    /// <description></description>
    [Description("ShareWeights"), MyTaskInfo(OneShot = false, Disabled = true)]
    public class MyShareWeightsTask : MyTask<MyAbstractWeightLayer>
    {
        [YAXSerializableField(DefaultValue = "")]
        [MyBrowsable, Category("\tSharing weights")]
        public String SourceNodeName { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("\tSharing weights")]
        public float ApproachRate { get; set; }

        private String m_previousSourceNodeName;
        private MyAbstractWeightLayer m_sourceLayer;
        private MyCudaKernel m_interpolateKernel;

        public MyShareWeightsTask() { } //parameterless constructor

        public override void Init(int nGPU)
        {
            m_interpolateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "Interpolate");
            m_previousSourceNodeName = "";
            m_sourceLayer = null;
        }

        private void FindSourceLayer()
        {
            m_sourceLayer = null;

            var matchingNodes = Owner.Owner.Network.GetChildNodesByName(SourceNodeName);

            if (matchingNodes.Count == 0)
            {
                MyLog.ERROR.WriteLine(Owner.Name + ": Cannot share weights with node " + SourceNodeName + " because it was not found!");
                return;
            }

            if (matchingNodes.Count > 1)
            {
                MyLog.ERROR.WriteLine(Owner.Name + ": Cannot share weights with node " + SourceNodeName + " because there are multiple nodes with this name!");
                return;
            }

            var sourceLayer = matchingNodes[0] as MyAbstractWeightLayer;

            if (sourceLayer == null)
            {
                MyLog.ERROR.WriteLine(Owner.Name + ": Cannot share weights with node " + SourceNodeName + " because it is not a weight layer!");
                return;                    
            }

            if (sourceLayer.Weights.Count != Owner.Weights.Count || sourceLayer.Bias.Count != Owner.Bias.Count)
            {
                MyLog.ERROR.WriteLine(Owner.Name + ": Cannot share weights with node " + SourceNodeName + " because the sizes do not match!");
                return;
            }

            if (sourceLayer.ActivationFunction != Owner.ActivationFunction)
            {
                MyLog.WARNING.WriteLine(Owner.Name + ": Sharing weights with node " + SourceNodeName + " but have a different activation function!");
            }

            m_sourceLayer = sourceLayer;
        }

        private void CopySourceLayerWeights()
        {
            if (m_sourceLayer != null)
            {
                m_sourceLayer.Weights.CopyToMemoryBlock(Owner.Weights, 0, 0, Owner.Weights.Count);
                m_sourceLayer.Bias.CopyToMemoryBlock(Owner.Bias, 0, 0, Owner.Bias.Count);
            }
        }

        public override void Execute()
        {
            if (SourceNodeName == "")
                return;

            if (SourceNodeName != m_previousSourceNodeName)
            {
                FindSourceLayer();
                CopySourceLayerWeights();
                m_previousSourceNodeName = SourceNodeName;
            }

            if (m_sourceLayer != null)
            {
                m_interpolateKernel.SetupExecution(Owner.Weights.Count);
                m_interpolateKernel.Run(Owner.Weights, m_sourceLayer.Weights, Owner.Weights, ApproachRate, Owner.Weights.Count);
                m_interpolateKernel.SetupExecution(Owner.Bias.Count);
                m_interpolateKernel.Run(Owner.Bias, m_sourceLayer.Bias, Owner.Bias, ApproachRate, Owner.Bias.Count);
            }
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
