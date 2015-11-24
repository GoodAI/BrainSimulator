using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.VSA;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <status>Working</status>
    /// <summary>Generates numbers from chosen distribution</summary>
    /// <description>The <b>SingleOutput</b> property will set random number on one (random) element of Output mem. block only.</description>
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

        public int Period { 
            get 
            { 
                return PeriodTask.Period;
            }
            set
            {
                PeriodTask.Period = value;
            }
        }

        public bool RandomPeriod { get { return PeriodTask.RandomPeriod; } }
        public int RandomPeriodMin { get { return PeriodTask.RandomPeriodMin; } }
        public int RandomPeriodMax { get { return PeriodTask.RandomPeriodMax; } }

        public MyMemoryBlock<float> RandomNumbers { get; private set; }

        public PeriodRNGTask PeriodTask { get; private set; }
        [MyTaskGroup("RNG")]
        public UniformRNGTask UniformRNG { get; private set; }
        [MyTaskGroup("RNG")]
        public NormalRNGTask NormalRNG { get; private set; }
        [MyTaskGroup("RNG")]
        public ConstantRNGTask ConstantRNG { get; private set; }
        [MyTaskGroup("RNG")]
        public CombinationRNGTask CombinationRNG { get; private set; }

        public Random m_rnd;
        public int NextPeriodChange;
        private MyCudaKernel m_setKernel;

        public MyRandomNode()
        {
            m_rnd = new Random(DateTime.Now.Millisecond);
            m_setKernel = MyKernelFactory.Instance.Kernel(GPU, @"Common\SetAllButOneKernel");
            NextPeriodChange = 1;
        }

        public override void OnSimulationStateChanged(Core.Execution.MySimulationHandler.StateEventArgs args)
        {
            base.OnSimulationStateChanged(args);

            if (args.NewState == Core.Execution.MySimulationHandler.SimulationState.STOPPED)
                NextPeriodChange = 1;
        }

        public override void UpdateMemoryBlocks()
        {
            RandomNumbers.Count = Output.Count + 1;

            if (UniformRNG.Enabled)
                UniformRNG.UpdateValueHints();
            else if (NormalRNG.Enabled)
                NormalRNG.UpdateValueHints();
            else if (ConstantRNG.Enabled)
                ConstantRNG.UpdateValueHints();
            else if (CombinationRNG.Enabled)
                CombinationRNG.UpdateValueHints();
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(RandomPeriodMin > 0, this, "RandomPeriodMin have to be greater than 0.");
            validator.AssertError(RandomPeriodMax > RandomPeriodMin, this, "RandomPeriodMax have to be greater than RandomPeriodMin");
            if(CombinationRNG.Enabled)
            {
                validator.AssertError(CombinationRNG.Min < CombinationRNG.Max, this, "Min has to be smaller than Max for Combination task.");
                validator.AssertError(Output.Count <= CombinationRNG.Max - CombinationRNG.Min, this, "Output is larger than Combination's task range.");
            }
        }

        public override string Description
        {
            get
            {
                if (UniformRNG.Enabled)
                    return UniformRNG.Description;
                else if (NormalRNG.Enabled)
                    return NormalRNG.Description;
                else if (ConstantRNG.Enabled)
                    return ConstantRNG.Description;
                else if (CombinationRNG.Enabled)
                    return CombinationRNG.Description;
                else
                    return base.Description;
            }
        }

        public void SetSingle()
        {
            if (SingleOutput == true)
            {
                //delete all values except one
                int keep = m_rnd.Next(Output.Count);
                m_setKernel.SetupExecution(Output.Count);
                m_setKernel.Run(Output, 0, keep, Output.Count);
            }
        }
        
        public void UpdatePeriod(MyTask caller)
        {
            if (RandomPeriod && (caller.SimulationStep >= NextPeriodChange))
            {
                //+1 for inclusive interval
                Period = m_rnd.Next(RandomPeriodMin, RandomPeriodMax + 1);
                NextPeriodChange = (int)caller.SimulationStep + Period;
            }
        }
    }


    /// <summary>This task holds information about period. Period information is in the task, so it can be changed during runtime.</summary>
    [Description("Period settings")]
    public class PeriodRNGTask: MyTask<MyRandomNode>
    {
        //Period should be randomized?
        [MyBrowsable, Category("\tPeriod parameters"), Description("Period will change randomly during simulation if set to True. RandomPeriodMin is a lower bound while RandomPeriodMax is an upper bound. First value is determined by Period parameter.")]
        [YAXSerializableField(DefaultValue = false)]
        public bool RandomPeriod { get; set; }

        //How often produce new numbers? 1 = every step
        //Also the first value for random period
        [MyBrowsable, Category("\tPeriod parameters"), Description("Number of simulation steps after which the random numbers will be regenerated. Also first value of period when RandomPeriod is True")]
        [YAXSerializableField(DefaultValue = 1)]
        public int Period { get; set; }

        //Minimum for random period - inclusive
        [MyBrowsable, Category("\tPeriod parameters"), Description("Lower bound for a random period when RandomPeriod is True")]
        [YAXSerializableField(DefaultValue = 1)]
        public int RandomPeriodMin { get; set; }

        //Maximum for random period - inclusive
        [MyBrowsable, Category("\tPeriod parameters"), Description("Upper bound for a random period when RandomPeriod is True")]
        [YAXSerializableField(DefaultValue = 10)]
        public int RandomPeriodMax { get; set; }

        public override void Init(int nGPU) { }

        public override void Execute()
        {
            
        }
    }


    /// <summary>Generates numbers from uniform distribution. Number will be in interval from MinValue to MaxValue.</summary>
    [Description("Uniform")]
    public class UniformRNGTask : MyTask<MyRandomNode>
    {
        private MyCudaKernel m_polynomialKernel;

        //Minimal value
        [MyBrowsable, Category("Uniform distribution"), DisplayName("M\tinValue"), Description("Lower bound for uniform distribution")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float MinValue { get; set; }

        //Maximum value
        [MyBrowsable, Category("Uniform distribution"), Description("Upper bound for uniform distribution")]
        [YAXSerializableField(DefaultValue = 1f)]
        public float MaxValue { get; set; }

        public void UpdateValueHints()
        {
            Owner.Output.MaxValueHint = MaxValue;
            Owner.Output.MinValueHint = MinValue;
        }

        public override void Init(int nGPU)
        {
            m_polynomialKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
        }

        public override void Execute()
        {
            if ((!Owner.RandomPeriod && (SimulationStep % Owner.Period == 0)) || (Owner.RandomPeriod && (SimulationStep == Owner.NextPeriodChange)))
            {
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.Output.GetDevice(Owner));

                //scale from 0-1 to min-max
                m_polynomialKernel.SetupExecution(Owner.Output.Count);
                m_polynomialKernel.Run(0, 0, (MaxValue - MinValue), MinValue,
                    Owner.Output,
                    Owner.Output,
                    Owner.Output.Count
                );

                Owner.SetSingle();
            }

            Owner.UpdatePeriod(this);
        }

        public string Description
        {
            get
            {
                return "Uniform (" + MinValue + "," + MaxValue + ")";
            }
        }
    }


    /// <summary>Generates numbers from normal distribution with parameters specified by Mean and StdDev.</summary>
    [Description("Normal")]
    public class NormalRNGTask : MyTask<MyRandomNode>
    {
        //Mean for normal dist.
        [MyBrowsable, Category("Normal distribution"), Description("Mean of the normal distribution")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float Mean { get; set; }

        //StdDev for normal dist.
        [MyBrowsable, Category("Normal distribution"), Description("Standard deviation of the normal distribution")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float StdDev { get; set; }

        public void UpdateValueHints()
        {
            Owner.Output.MaxValueHint = 3 * StdDev;
            Owner.Output.MinValueHint = -3 * StdDev;
        }

        public override void Init(int nGPU){ }

        public override void Execute()
        {
            if ((!Owner.RandomPeriod && (SimulationStep % Owner.Period == 0)) || (Owner.RandomPeriod && (SimulationStep == Owner.NextPeriodChange)))
            {
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

                Owner.SetSingle();
            }

            Owner.UpdatePeriod(this);
        }

        public string Description
        {
            get
            {
                return "Normal (" + Mean + "," + StdDev + ")";
            }
        }
    }


    /// <summary>Generates constant number.</summary>
    [Description("Constant")]
    public class ConstantRNGTask : MyTask<MyRandomNode>
    {
        private MyCudaKernel m_kernel;

        //Constant value
        [MyBrowsable, Category("Constant distribution"), Description("Which number to generate")]
        [YAXSerializableField(DefaultValue = 1f)]
        public float Constant { get; set; }

        public void UpdateValueHints()
        {
            Owner.Output.MinValueHint = Constant;
            Owner.Output.MaxValueHint = Constant;
        }

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Predictions\ResetLayerKernel");
        }

        public override void Execute()
        {
            if ((!Owner.RandomPeriod && (SimulationStep % Owner.Period == 0)) || (Owner.RandomPeriod && (SimulationStep == Owner.NextPeriodChange)))
            {
                m_kernel.SetupExecution(Owner.Output.Count);
                m_kernel.Run(Owner.Output, Constant, Owner.Output.Count);

                Owner.SetSingle();
            }

            Owner.UpdatePeriod(this);
        }

        public string Description
        {
            get
            {
                return "Constant (" + Constant + ")";
            }
        }
    }

    /// <summary>Generates (possibly unique) integers between Min and Max where Min is inclusive and Max is exclusive.</summary>
    [Description("Combination")]
    public class CombinationRNGTask : MyTask<MyRandomNode>
    {
        //Minimal value
        [MyBrowsable, Category("Combinations"), DisplayName("M\tin"), Description("Minimum (inclusive) number to be generated")]
        [YAXSerializableField(DefaultValue = 0)]
        public int Min { get; set; }

        //Maximum value
        [MyBrowsable, Category("Combinations"), Description("Maximum (exclusive) number to be generated.")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int Max { get; set; }

        //Maximum value
        [MyBrowsable, Category("Combinations"), Description("Numbers will be unique if set to True")]
        [YAXSerializableField(DefaultValue = true)]
        public bool Unique { get; set; }

        private HashSet<float> combHS;

        public void UpdateValueHints()
        {
            Owner.Output.MaxValueHint = Max;
            Owner.Output.MinValueHint = Min;
        }

        public override void Init(int nGPU)
        {
            combHS = new HashSet<float>();
        }

        public override void Execute()
        {
            if ((!Owner.RandomPeriod && (SimulationStep % Owner.Period == 0)) || (Owner.RandomPeriod && (SimulationStep == Owner.NextPeriodChange)))
            {
                if (!Owner.Output.OnHost)
                    Owner.Output.SafeCopyToHost();

                if (Unique)
                    MyCombinationBase.GenerateCombinationUnique(new ArraySegment<float>(Owner.Output.Host), combHS, Min, Max, Owner.m_rnd);
                else
                    MyCombinationBase.GenerateCombination(new ArraySegment<float>(Owner.Output.Host), Min, Max, Owner.m_rnd);

                Owner.Output.SafeCopyToDevice();

                Owner.SetSingle();
            }

            Owner.UpdatePeriod(this);
        }

        public string Description
        {
            get
            {
                return "Combination [" + Min + "," + Max + ")" + MyCombinationBook.GetPowerString(Owner.OutputSize);
            }
        }
    }
}

