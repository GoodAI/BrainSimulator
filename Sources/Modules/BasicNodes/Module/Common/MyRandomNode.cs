using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Core.Task;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading.Tasks;
using GoodAI.Modules.VSA;
using YAXLib;
using ManagedCuda;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <status>Working</status>
    /// <summary>Generates numbers from chosen distribution</summary>
    /// <description></description>
    public class MyRandomNode : MyWorkingNode
    {
        //Only one of outputs should be active each time. Maybe generalize to arbitrary number?
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = false)]
        public bool SingleOutput { get; set; }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("IO")]
        public int OutputSize
        {
            get { return Output.Count; }
            set { Output.Count = value; }
        }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1)]
        public int ColumnHint
        {
            get
            {
                return Output != null ? Output.ColumnHint : 0;
            }
            set
            {
                if (Output != null)
                    Output.ColumnHint = value;
            }
        }


        public MyMemoryBlock<float> RandomNumbers { get; private set; }

        public MyRNGTask MakeDecision { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            RandomNumbers.Count = Output.Count + 1;
            MakeDecision.UpdateValueHints();
        }

        public override string Description
        {
            get
            {
                return MakeDecision.Enabled ? MakeDecision.Description : base.Description;
            }
        }
    }

    [Description("Generate random numbers")]
    public class MyRNGTask : MyTask<MyRandomNode>
    {
        private Random m_rnd;
        private int m_nextPeriodChange;
        private MyCudaKernel m_kernel;
        private MyCudaKernel m_setKernel;
        private MyCudaKernel m_polynomialKernel;

        public enum RandomDistribution
        {
            Uniform,
            Normal,
            Constant,
            Combination,
        }

        //Choose distribution
        [MyBrowsable, Category("\tParams")]
        [YAXSerializableField(DefaultValue = RandomDistribution.Uniform)]
        public RandomDistribution Distribution { get; set; }

        //Period should be randomized?
        [MyBrowsable, Category("\tPeriod parameters")]
        [YAXSerializableField(DefaultValue = false)]
        public bool RandomPeriod { get; set; }

        //How often produce new numbers? 1 = every step
        //Also the first value for random period
        [MyBrowsable, Category("\tPeriod parameters")]
        [YAXSerializableField(DefaultValue = 1)]
        public int Period { get; set; }

        //Minimum for random period - inclusive
        [MyBrowsable, Category("\tPeriod parameters")]
        [YAXSerializableField(DefaultValue = 1)]
        public int RandomPeriodMin { get; set; }

        //Maximum for random period - inclusive
        [MyBrowsable, Category("\tPeriod parameters")]
        [YAXSerializableField(DefaultValue = 10)]
        public int RandomPeriodMax { get; set; }

        //Minimal value
        [MyBrowsable, Category("Uniform distribution"), DisplayName("M\tinValue")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float MinValue { get; set; }

        //Maximum value
        [MyBrowsable, Category("Uniform distribution")]
        [YAXSerializableField(DefaultValue = 1f)]
        public float MaxValue { get; set; }

        //Mean for normal dist.
        [MyBrowsable, Category("Normal distribution")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float Mean { get; set; }

        //StdDev for normal dist.
        [MyBrowsable, Category("Normal distribution")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float StdDev { get; set; }

        //Constant value
        [MyBrowsable, Category("Constant distribution")]
        [YAXSerializableField(DefaultValue = 1f)]
        //public float Constant { get; set; }
        public float Constant { get; set; }

        //Minimal value
        [MyBrowsable, Category("Combinations"), DisplayName("M\tin")]
        [YAXSerializableField(DefaultValue = 0)]
        public int Min { get; set; }

        //Maximum value
        [MyBrowsable, Category("Combinations")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int Max { get; set; }

        //Maximum value
        [MyBrowsable, Category("Combinations")]
        [YAXSerializableField(DefaultValue = true)]
        public bool Unique { get; set; }

        private HashSet<float> combHS;


        public void UpdateValueHints()
        {
            //if SingleOutput is TRUE, than zero values wont be probably black
            switch (Distribution)
            {
                case RandomDistribution.Constant:
                    Owner.Output.MinValueHint = Constant;
                    Owner.Output.MaxValueHint = Constant;
                    break;
                case RandomDistribution.Uniform:
                    Owner.Output.MaxValueHint = MaxValue;
                    Owner.Output.MinValueHint = MinValue;
                    break;
                case RandomDistribution.Normal: //set to three-sigma ~ covers 93.3% values
                    Owner.Output.MaxValueHint = 3 * StdDev;
                    Owner.Output.MinValueHint = -3 * StdDev;
                    break;
                case RandomDistribution.Combination:
                    Owner.Output.MaxValueHint = Max;
                    Owner.Output.MinValueHint = Min;
                    break;
            }
        }

        public override void Init(int nGPU)
        {
            m_rnd = new Random(DateTime.Now.Millisecond);
            m_nextPeriodChange = 1;
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Predictions\ResetLayerKernel");
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetAllButOneKernel");
            m_polynomialKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");

            //validation
            if (RandomPeriodMin < 1)
            {
                MyLog.WARNING.WriteLine("RandomPeriodMin have to be greater than 1. Once it is generated negative, no new Periods will be generated.");
            }
            if (RandomPeriodMax < RandomPeriodMin)
            {
                MyLog.ERROR.WriteLine("RandomPeriodMax have to be greater than RandomPeriodMin. This is about to crash.");
            }

            combHS = new HashSet<float>();
        }

        public override void Execute()
        {
            if ((!RandomPeriod && (SimulationStep % Period == 0)) || (RandomPeriod && (SimulationStep == m_nextPeriodChange)))
            {
                switch (Distribution)
                {
                    case RandomDistribution.Constant:
                        m_kernel.SetupExecution(Owner.Output.Count);
                        m_kernel.Run(Owner.Output, Constant, Owner.Output.Count);
                        break;
                    case RandomDistribution.Uniform:
                        MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.Output.GetDevice(Owner));
                        break;
                    case RandomDistribution.Normal:
                        //Normal RNG uses Box-Muller transformation, so it can generate only even number of values
                        if (Owner.Output.Count % 2 == 0)
                        {
                            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Output.GetDevice(Owner), Mean, StdDev);
                        }
                        else
                        {
                            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.RandomNumbers.GetDevice(Owner), Mean, StdDev);
                            Owner.RandomNumbers.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.Output.Count);
                        }

                        break;

                    case RandomDistribution.Combination:
                        if (!Owner.Output.OnHost)
                            Owner.Output.SafeCopyToHost();

                        if (Unique)
                            MyCombinationBase.GenerateCombinationUnique(new ArraySegment<float>(Owner.Output.Host), combHS, Min, Max, m_rnd);
                        else
                            MyCombinationBase.GenerateCombination(new ArraySegment<float>(Owner.Output.Host), Min, Max, m_rnd);

                        Owner.Output.SafeCopyToDevice();
                        break;
                }

                if (Distribution == RandomDistribution.Uniform)
                {
                    //scale from 0-1 to min-max
                    m_polynomialKernel.SetupExecution(Owner.Output.Count);
                    m_polynomialKernel.Run(0, 0, (MaxValue - MinValue), MinValue,
                        Owner.Output,
                        Owner.Output,
                        Owner.Output.Count
                    );
                }

                if (Owner.SingleOutput == true)
                {
                    //delete all values except one
                    int keep = m_rnd.Next(Owner.Output.Count);
                    m_setKernel.SetupExecution(Owner.Output.Count);
                    m_setKernel.Run(Owner.Output, 0, keep, Owner.Output.Count);
                }
            }
            if (RandomPeriod && (SimulationStep == m_nextPeriodChange))
            {
                //+1 for inclusive interval
                Period = m_rnd.Next(RandomPeriodMin, RandomPeriodMax + 1);
                m_nextPeriodChange += Period;
            }
        }

        public string Description
        {
            get
            {
                string ret = Distribution.ToString() + " ";
                switch (Distribution)
                {
                    case RandomDistribution.Constant:
                        ret += "(" + Constant + ")";
                        break;
                    case RandomDistribution.Normal:
                        ret += "(" + Mean + "," + StdDev + ")";
                        break;
                    case RandomDistribution.Uniform:
                        ret += "(" + MinValue + "," + MaxValue + ")";
                        break;
                    case RandomDistribution.Combination:
                        ret += "[" + Min + "," + Max + ")";
                        break;
                }

                return ret + MyCombinationBook.GetPowerString(Owner.OutputSize);
            }
        }
    }
}

