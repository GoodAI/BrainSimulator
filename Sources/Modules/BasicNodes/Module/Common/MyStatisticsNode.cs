using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System.Linq;
using System.ComponentModel;
using System.Diagnostics;
using YAXLib;
using System.Collections.Generic;
using System;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>ms</meta>
    /// <status>Working</status>
    /// <summary>Calculates statistical paramters.</summary>
    /// <description></description>
    public class MyStatisticsNode : MyWorkingNode
    {
        #region inputs
        [MyBrowsable, Category("Input interface")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int INPUT_SIZE { get; private set; }

        [MyBrowsable, Category("History")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("History")]
        public int WindowLength { get; set; }

        [MyBrowsable, Category("I/O")]
        [YAXSerializableField(DefaultValue = 1)]
        public int ColumnHint
        {
            get
            {
                return this.Output != null ? Output.ColumnHint : 1;
            }
            set
            {
                if (this.Output != null) this.Output.ColumnHint = value;
                if (value < 1) this.Output.ColumnHint = 1;
            }
        }

        public enum CalculateForEnum { AllElements, Rows, Columns, WholeMatrix }
        [MyBrowsable, Category("\tConsolidation"), Description(" ")]
        [YAXSerializableField(DefaultValue = 0)]
        public CalculateForEnum CalculateFor { get; set; }
        #endregion

        #region memoryblocks
        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyPersistable]
        public MyMemoryBlock<float> Window { get; private set; }

        public MyMemoryBlock<float> Temp { get; private set; }

        #endregion

        #region tasks
        [MyTaskGroup("G")]
        public MeanTask Mean { get; private set; }
        [MyTaskGroup("G")]
        public VarianceTask Variance { get; private set; }
        [MyTaskGroup("G")]
        public ModeTask Mode { get; private set; }
        #endregion

        public override void UpdateMemoryBlocks()
        {
            if (Input == null) return;

            int InputRowN = INPUT_SIZE / Input.ColumnHint;
            int InputColumnN = Input.ColumnHint;

            if (CalculateFor == CalculateForEnum.AllElements) Output.Count = INPUT_SIZE;
            else if (CalculateFor == CalculateForEnum.Rows) Output.Count = InputRowN;
            else if (CalculateFor == CalculateForEnum.Columns) Output.Count = InputColumnN;
            else if (CalculateFor == CalculateForEnum.WholeMatrix) Output.Count = 1;
            Window.Count = Output.Count * WindowLength;
            Temp.Count = Output.Count;
        }

        public override string Description
        {
            get
            {
                if (Mean.Enabled) { return Mean.Description; }
                if (Variance.Enabled) { return Variance.Description; }
                else return base.Description;
            }
        }


        /// <summary>Returns mean for each element of matrix through time window, row, column or total. If used with Window > 1, first process rows/columns/all, then process over window.</summary>
        [Description("Mean")]
        public class MeanTask : MyTask<MyStatisticsNode>
        {
            private int WindowIdx { get; set; }
            private int Base { get; set; }
            private int ColumnsN, RowsN, ElementsN;
            private float[] ActualConslMeans;
            MyCudaKernel m_sumKernel;
            private readonly int HOST_DEVICE_THRSHD = 1 << 13;

            public override void Init(int nGPU)
            {
                WindowIdx = 0;
                Base = 0;
                ColumnsN = Owner.Input.ColumnHint;
                RowsN = Owner.INPUT_SIZE / Owner.Input.ColumnHint;
                ColumnsN = Owner.Input.ColumnHint;
                ElementsN = Owner.INPUT_SIZE;
                ActualConslMeans = new float[Owner.Output.Count];
                m_sumKernel = MyReductionFactory.Kernel(nGPU, MyReductionFactory.Mode.f_Sum_f);
            }

            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();
                // Owner.Window.SafeCopyToHost();

                // Calculate vals for actual Matrix
                switch (Owner.CalculateFor)
                {
                    case CalculateForEnum.AllElements:
                        for (int i = 0; i < Owner.Output.Count; i++)
                        {
                            ActualConslMeans[i] = Owner.Input.Host[i];
                        }
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < ColumnsN; i++)
                        {
                            ActualConslMeans[i] = Mean(Owner.Input, Owner.Temp, i, ColumnsN, RowsN);
                        }
                        break;
                    case CalculateForEnum.Rows:
                        for (int i = 0; i < RowsN; i++)
                        {
                            ActualConslMeans[i] = Mean(Owner.Input, Owner.Temp, i * ColumnsN, 1, ColumnsN);
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        ActualConslMeans[0] = Mean(Owner.Input, Owner.Temp, 0, 1, RowsN * ColumnsN);
                        break;
                }

                // For whole Window
                for (int i = 0; i < Owner.Output.Count; i++, WindowIdx++)
                {
                    if (Base == Owner.WindowLength)
                    {
                        Owner.Output.Host[i] = iterativeMeanSub(Owner.Output.Host[i], Owner.Window.Host[WindowIdx], Base--);
                        Owner.Output.Host[i] = iterativeMeanAdd(Owner.Output.Host[i], ActualConslMeans[i], Base++);
                    }
                    else
                    {
                        Owner.Output.Host[i] = iterativeMeanAdd(Owner.Output.Host[i], ActualConslMeans[i], Base);
                    }


                    Owner.Window.Host[WindowIdx] = ActualConslMeans[i];
                }
                WindowIdx %= Owner.Window.Count;
                if (Base < Owner.WindowLength) Base++;


                Owner.Window.SafeCopyToDevice(); // is it appropriate?
                Owner.Output.SafeCopyToDevice();
            }

            public float iterativeMeanAdd(float Mean, float NewValue, int totalElems)
            {
                Mean += ((NewValue - Mean) / (totalElems + 1));
                return Mean;
            }

            public float iterativeMeanSub(float Mean, float OldValue, int totalElems)
            {
                if (totalElems == 1) return 0;
                Mean += ((Mean - OldValue) / (totalElems - 1));
                return Mean;
            }

            public float Mean(MyMemoryBlock<float> Input, MyMemoryBlock<float> Temp, int StartIndex, int Step, int StepN)
            {
                float f = 0;
                if (StepN >= HOST_DEVICE_THRSHD)
                {
                    m_sumKernel.Run(Temp, Input, StepN * Step, 0, StartIndex, Step, /* distributed: */ 0);
                    Owner.Temp.SafeCopyToHost();
                    f = Owner.Temp.Host[0] / StepN;
                    return f;
                }

                for (int i = 0; i < StepN; i++)
                {
                    int Idx = StartIndex + i * Step;
                    f += Input.Host[StartIndex + i * Step];
                }
                f /= StepN;
                return f;
            }

            public string Description
            {
                get
                {
                    switch (Owner.CalculateFor)
                    {
                        case CalculateForEnum.AllElements: return "Means of elements";
                        case CalculateForEnum.Rows: return "Means of rows";
                        case CalculateForEnum.Columns: return "Means of columns";
                        case CalculateForEnum.WholeMatrix: return "Overall mean";
                        default: MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Using MeanTask with unknown parameter"); return "";
                    }
                }
            }
        }

        /// <summary>Returns variances for rows, column or total. If used with Window > 1, first process rows/columns/all, then process over window.</summary>
        [Description("Variance")]
        public class VarianceTask : MyTask<MyStatisticsNode>
        {
            private int WindowIdx { get; set; }
            private int Base { get; set; }
            private int ColumnsN, RowsN, ElementsN;
            private float[] ActualConslVals;
            private float[] Ks;
            private float[] Exs;
            private float[] Ex2s;

            public override void Init(int nGPU)
            {
                WindowIdx = 0;
                ColumnsN = Owner.Input.ColumnHint;
                RowsN = Owner.INPUT_SIZE / Owner.Input.ColumnHint;
                ElementsN = Owner.INPUT_SIZE;

                Base = 0;

                ActualConslVals = new float[Owner.Output.Count];
                Ks = new float[Owner.Output.Count];
                Exs = new float[Owner.Output.Count];
                Ex2s = new float[Owner.Output.Count];
            }

            public override void Execute()
            {
                Owner.Input.SafeCopyToHost();
                //Owner.Window.SafeCopyToHost();

                // calculate vals for actual matrix
                switch (Owner.CalculateFor)
                {
                    case CalculateForEnum.AllElements:
                        for (int i = 0; i < Owner.Output.Count; i++)
                        {
                            ActualConslVals[i] = Owner.Input.Host[i];
                        }
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < ColumnsN; i++)
                        {
                            ActualConslVals[i] = Varience(Owner.Input.Host, i, ColumnsN, RowsN);
                        }
                        break;
                    case CalculateForEnum.Rows:
                        for (int i = 0; i < RowsN; i++)
                        {
                            ActualConslVals[i] = Varience(Owner.Input.Host, i * ColumnsN, 1, ColumnsN);
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        ActualConslVals[0] = Varience(Owner.Input.Host, 0, 1, ColumnsN * RowsN);
                        break;
                }

                if (Owner.WindowLength == 1)
                {
                    ActualConslVals.CopyTo(Owner.Window.Host, 0);
                    ActualConslVals.CopyTo(Owner.Output.Host, 0);
                }
                else
                {
                    // calculate for whole window
                    for (int i = 0; i < Owner.Output.Count; i++, WindowIdx++)
                    {
                        if (Base == Owner.WindowLength)
                        {
                            Owner.Output.Host[i] = iterativeVarianceSub(Owner.Window.Host[WindowIdx], i, Base--);
                            Owner.Output.Host[i] = iterativeVarianceAdd(ActualConslVals[i], i, Base++);
                        }
                        else
                        {
                            Owner.Output.Host[i] = iterativeVarianceAdd(ActualConslVals[i], i, Base);
                        }
                        Owner.Window.Host[WindowIdx] = ActualConslVals[i];
                    }
                }

                WindowIdx %= Owner.Window.Count;
                if (Base < Owner.WindowLength) Base++;


                Owner.Window.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }

            public float Varience(float[] Input, int StartIndex, int Step, int StepN)
            {
                float K = Input[StartIndex];
                float Ex = 0;
                float Ex2 = 0;
                for (int i = 0; i < Step * StepN; i += Step)
                {
                    float newVal = Input[StartIndex + i];
                    Ex += newVal - K;
                    Ex2 += (newVal - K) * (newVal - K);
                }
                float f = (Ex2 - (Ex * Ex) / StepN) / (StepN - 1);
                return (float) f;
            }

            public float iterativeVarianceAdd(float NewValue, int Idx, int Base)
            {
                float DNewValue = (float) NewValue;
                if (Base == 0) Ks[Idx] = DNewValue;
                Exs[Idx] += DNewValue - Ks[Idx];
                Ex2s[Idx] += (DNewValue - Ks[Idx]) * (DNewValue - Ks[Idx]);
                return getIterVariance(Idx, Base + 1);
            }

            public float iterativeVarianceSub(float OldValue, int Idx, int Base)
            {
                float DOldValue = (float) OldValue;
                Exs[Idx] -= DOldValue - Ks[Idx];
                Ex2s[Idx] -= (DOldValue - Ks[Idx]) * (DOldValue - Ks[Idx]);
                return getIterVariance(Idx, Base - 1);
            }

            public float getIterVariance(int Idx, int Base)
            {
                Debug.Assert(Base >= 0, "getIterVarience called with negative Base");
                if (Base == 0) return float.NaN;
                if (Base == 1) return float.NaN;
                float var = (Ex2s[Idx] - (Exs[Idx] * Exs[Idx]) / (float) Base) / (float) (Base - 1);
                return (float) var;
            }

            public float getIterMean(int Idx, int Base)
            {
                Debug.Assert(Base >= 0, "getIterVarience called with negative Base");
                if (Base == 0) return float.NaN;
                if (Base == 1) return float.NaN;
                float var = Ks[Idx] + Exs[Idx] / (float) Base;
                return (float) var;
            }

            public string Description
            {
                get
                {
                    switch (Owner.CalculateFor)
                    {
                        case CalculateForEnum.AllElements: return "Variance of elements";
                        case CalculateForEnum.Rows: return "Variance of rows";
                        case CalculateForEnum.Columns: return "Variance of columns";
                        case CalculateForEnum.WholeMatrix: return "Overall variance";
                        default: MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Using VarianceTask with unknown parameter"); return "";
                    }
                }
            }
        }


        [Description("Mode")]
        /// <summary>Returns the most frequent value for every element through time window, rows, columns or total.
        /// If used with Window > 1, first process rows/columns/all, then process over window.</summary>
        public class ModeTask : MyTask<MyStatisticsNode>
        {
            public enum MaxMin { Max, Min }
            [MyBrowsable, Category("Params"), Description("If more values have same frequency, return max/min of them")]
            [YAXSerializableField(DefaultValue = MaxMin.Max)]
            public MaxMin ReturnValue { get; set; }

            private int WindowIdx { get; set; }
            private int Base { get; set; }
            private int ColumnsN, RowsN, ElementsN;
            private float[] ActualConslVals;

            delegate float MaxMinFunction(IEnumerable<float> source);

            public override void Init(int nGPU)
            {
                WindowIdx = 0;
                Base = 0;
                ColumnsN = Owner.Input.ColumnHint;
                RowsN = Owner.INPUT_SIZE / Owner.Input.ColumnHint;
                ColumnsN = Owner.Input.ColumnHint;
                ElementsN = Owner.INPUT_SIZE;
                ActualConslVals = new float[Owner.Output.Count];
            }


            public override void Execute()
            {
                MaxMinFunction mmf;
                if (ReturnValue == MaxMin.Max) mmf = Enumerable.Max;
                else if (ReturnValue == MaxMin.Min) mmf = Enumerable.Min;
                else { Debug.Assert(false, "Unknow parameter in StatisticsNode, ModeTask, ReturnValue"); return; }

                Owner.Input.SafeCopyToHost();
                Owner.Window.SafeCopyToHost();

                // calculate vals for actual matrix
                float[] Input = Owner.Input.Host;
                switch (Owner.CalculateFor)
                {
                    case CalculateForEnum.AllElements:
                        ActualConslVals = Input;
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < ColumnsN; i++)
                        {
                            ActualConslVals[i] = mmf(Mode(Input, i, ColumnsN, RowsN));
                        }
                        break;
                    case CalculateForEnum.Rows:
                        for (int i = 0; i < RowsN; i++)
                        {
                            ActualConslVals[i] = mmf(Mode(Input, i * ColumnsN, 1, ColumnsN));
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        ActualConslVals[0] = mmf(Mode(Input));
                        break;
                }

                if (Owner.WindowLength == 1)
                {
                    ActualConslVals.CopyTo(Owner.Window.Host, 0);
                    ActualConslVals.CopyTo(Owner.Output.Host, 0);
                }
                else
                {
                    // calculate for whole window
                    if (Base < Owner.WindowLength) Base++;
                    for (int i = 0; i < Owner.Output.Count; i++, WindowIdx++)
                    {
                        Owner.Window.Host[WindowIdx] = ActualConslVals[i];
                        Owner.Output.Host[i] = mmf(Mode(Owner.Window.Host, i, Owner.Output.Count, Base));
                    }
                }
                WindowIdx %= Owner.Window.Count;

                Owner.Window.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }

            public List<float> Mode(float[] f, int StartIdx, int Step, int StepN)
            {
                float[] fm = new float[StepN];

                for (int j = 0, i = StartIdx; j < StepN; j++, i += Step)
                {
                    fm[j] = f[i];
                }

                return Mode(fm);
            }

            public List<float> Mode(float[] f)
            {
                var groups = f.GroupBy(v => v);
                int maxCount = groups.Max(g => g.Count());
                float mode = groups.First(g => g.Count() == maxCount).Key;
                var maxGroups = groups.Where(g => g.Count() == maxCount);
                List<float> r = new List<float>();
                foreach (var item in maxGroups) r.Add(item.Key);
                return r;
            }

            public string Description
            {
                get
                {
                    switch (Owner.CalculateFor)
                    {
                        case CalculateForEnum.AllElements: return "Mode of elements";
                        case CalculateForEnum.Rows: return "Mode of rows";
                        case CalculateForEnum.Columns: return "Mode of columns";
                        case CalculateForEnum.WholeMatrix: return "Overall Mode";
                        default: MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Using ModeTask with unknown parameter"); return "";
                    }
                }
            }
        }
    }
}
