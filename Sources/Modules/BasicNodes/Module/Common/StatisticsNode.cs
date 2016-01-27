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
    public class StatisticsNode : MyWorkingNode
    {
        #region inputs
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

            int inputRowN = Input.Count / Input.ColumnHint;
            int inputColumnN = Input.ColumnHint;

            if (CalculateFor == CalculateForEnum.AllElements) Output.Count = Input.Count;
            else if (CalculateFor == CalculateForEnum.Rows) Output.Count = inputRowN;
            else if (CalculateFor == CalculateForEnum.Columns) Output.Count = inputColumnN;
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
                if (Mode.Enabled) { return Mode.Description; }
                else return base.Description;
            }
        }


        /// <summary>Returns mean for each element of matrix through time window, row, column or total.
        /// If used with Window > 1, first process rows/columns/all, then process over window.</summary>
        [Description("Mean")]
        public class MeanTask : MyTask<StatisticsNode>
        {
            private int windowIdx { get; set; }
            private int elementselementsCountN { get; set; }
            private int columnsN, rowsN, elementsN;
            private float[] m_actualConslMeans;
            MyReductionKernel<float> m_sumKernel;
            private const int HostToDeviceThreshold = 1 << 13;

            public override void Init(int nGPU)
            {
                windowIdx = 0;
                elementselementsCountN = 0;
                columnsN = Owner.Input.ColumnHint;
                rowsN = Owner.Input.Count / Owner.Input.ColumnHint;
                columnsN = Owner.Input.ColumnHint;
                elementsN = Owner.Input.Count;
                m_actualConslMeans = new float[Owner.Output.Count];
                m_sumKernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);
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
                            m_actualConslMeans[i] = Owner.Input.Host[i];
                        }
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < columnsN; i++)
                        {
                            m_actualConslMeans[i] = Mean(Owner.Input, Owner.Temp, i, columnsN, rowsN);
                        }
                        break;
                    case CalculateForEnum.Rows:
                        for (int i = 0; i < rowsN; i++)
                        {
                            m_actualConslMeans[i] = Mean(Owner.Input, Owner.Temp, i * columnsN, 1, columnsN);
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        m_actualConslMeans[0] = Mean(Owner.Input, Owner.Temp, 0, 1, rowsN * columnsN);
                        break;
                }

                // For whole Window
                for (int i = 0; i < Owner.Output.Count; i++, windowIdx++)
                {
                    if (elementselementsCountN == Owner.WindowLength)
                    {
                        Owner.Output.Host[i] = IterativeMeanSub(Owner.Output.Host[i], Owner.Window.Host[windowIdx], elementselementsCountN--);
                        Owner.Output.Host[i] = IterativeMeanAdd(Owner.Output.Host[i], m_actualConslMeans[i], elementselementsCountN++);
                    }
                    else
                    {
                        Owner.Output.Host[i] = IterativeMeanAdd(Owner.Output.Host[i], m_actualConslMeans[i], elementselementsCountN);
                    }


                    Owner.Window.Host[windowIdx] = m_actualConslMeans[i];
                }
                windowIdx %= Owner.Window.Count;
                if (elementselementsCountN < Owner.WindowLength) elementselementsCountN++;


                Owner.Window.SafeCopyToDevice(); // is it appropriate?
                Owner.Output.SafeCopyToDevice();
            }

            public float IterativeMeanAdd(float actualMean, float newValue, int totalElems)
            {
                actualMean += ((newValue - actualMean) / (totalElems + 1));
                return actualMean;
            }

            public float IterativeMeanSub(float actualMean, float oldValue, int totalElems)
            {
                if (totalElems == 1) return 0;
                actualMean += ((actualMean - oldValue) / (totalElems - 1));
                return actualMean;
            }

            public float Mean(MyMemoryBlock<float> values, MyMemoryBlock<float> temporalMemory, int startIndex, int step, int stepsN)
            {
                float f = 0;
                if (stepsN >= HostToDeviceThreshold)
                {
                    //ZXC m_sumKernel.Run(temporalMemory, values, stepsN * step, 0, startIndex, step, /* distributed: */ 0);
                    m_sumKernel.size = stepsN * step;
                    m_sumKernel.inOffset = startIndex;
                    m_sumKernel.stride = step;
                    m_sumKernel.Run(temporalMemory, values);
                    Owner.Temp.SafeCopyToHost();
                    f = Owner.Temp.Host[0] / stepsN;
                    return f;
                }

                for (int i = 0; i < stepsN; i++)
                {
                    int Idx = startIndex + i * step;
                    f += values.Host[startIndex + i * step];
                }
                f /= stepsN;
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
        public class VarianceTask : MyTask<StatisticsNode>
        {
            private int windowIdx { get; set; }
            private int elementselementsCountN { get; set; }
            private int columnsN, rowsN, elementsN;
            private float[] m_actualConslVals;
            private float[] m_ks;
            private float[] m_exs;
            private float[] m_ex2s;

            public override void Init(int nGPU)
            {
                windowIdx = 0;
                columnsN = Owner.Input.ColumnHint;
                rowsN = Owner.Input.Count / Owner.Input.ColumnHint;
                elementsN = Owner.Input.Count;

                elementselementsCountN = 0;

                m_actualConslVals = new float[Owner.Output.Count];
                m_ks = new float[Owner.Output.Count];
                m_exs = new float[Owner.Output.Count];
                m_ex2s = new float[Owner.Output.Count];
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
                            m_actualConslVals[i] = Owner.Input.Host[i];
                        }
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < columnsN; i++)
                        {
                            m_actualConslVals[i] = Varience(Owner.Input.Host, i, columnsN, rowsN);
                        }
                        break;
                    case CalculateForEnum.Rows:
                        for (int i = 0; i < rowsN; i++)
                        {
                            m_actualConslVals[i] = Varience(Owner.Input.Host, i * columnsN, 1, columnsN);
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        m_actualConslVals[0] = Varience(Owner.Input.Host, 0, 1, columnsN * rowsN);
                        break;
                }

                if (Owner.WindowLength == 1)
                {
                    m_actualConslVals.CopyTo(Owner.Window.Host, 0);
                    m_actualConslVals.CopyTo(Owner.Output.Host, 0);
                }
                else
                {
                    // calculate for whole window
                    for (int i = 0; i < Owner.Output.Count; i++, windowIdx++)
                    {
                        if (elementselementsCountN == Owner.WindowLength)
                        {
                            Owner.Output.Host[i] = IterativeVarianceSub(Owner.Window.Host[windowIdx], i, elementselementsCountN--);
                            Owner.Output.Host[i] = IterativeVarianceAdd(m_actualConslVals[i], i, elementselementsCountN++);
                        }
                        else
                        {
                            Owner.Output.Host[i] = IterativeVarianceAdd(m_actualConslVals[i], i, elementselementsCountN);
                        }
                        Owner.Window.Host[windowIdx] = m_actualConslVals[i];
                    }
                }

                windowIdx %= Owner.Window.Count;
                if (elementselementsCountN < Owner.WindowLength) elementselementsCountN++;


                Owner.Window.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }

            public float Varience(float[] values, int startIndex, int step, int stepsN)
            {
                float k = values[startIndex];
                float ex = 0;
                float ex2 = 0;
                for (int i = 0; i < step * stepsN; i += step)
                {
                    float newVal = values[startIndex + i];
                    ex += newVal - k;
                    ex2 += (newVal - k) * (newVal - k);
                }
                float f = (ex2 - (ex * ex) / stepsN) / (stepsN - 1);
                return (float) f;
            }

            public float IterativeVarianceAdd(float newValue, int idx, int elementsCountN)
            {
                if (elementsCountN == 0) m_ks[idx] = newValue;
                m_exs[idx] += newValue - m_ks[idx];
                m_ex2s[idx] += (newValue - m_ks[idx]) * (newValue - m_ks[idx]);
                return CalculateIterativeVariance(idx, elementsCountN + 1);
            }

            public float IterativeVarianceSub(float oldValue, int idx, int elementsCountN)
            {
                m_exs[idx] -= oldValue - m_ks[idx];
                m_ex2s[idx] -= (oldValue - m_ks[idx]) * (oldValue - m_ks[idx]);
                return CalculateIterativeVariance(idx, elementsCountN - 1);
            }

            public float CalculateIterativeVariance(int idx, int elementsCountN)
            {
                Debug.Assert(elementsCountN >= 0, "getIterVarience called with negative elementsCountN");
                if (elementsCountN == 0) return float.NaN;
                if (elementsCountN == 1) return m_ks[0];
                float var = (m_ex2s[idx] - (m_exs[idx] * m_exs[idx]) / (float) elementsCountN) / (float) (elementsCountN - 1);
                return (float) var;
            }

            public float CalculateIterativeMean(int idx, int elementsCountN)
            {
                Debug.Assert(elementsCountN >= 0, "getIterVarience called with negative elementsCountN");
                if (elementsCountN == 0) return float.NaN;
                if (elementsCountN == 1) return m_ks[0];
                float var = m_ks[idx] + m_exs[idx] / (float) elementsCountN;
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
        public class ModeTask : MyTask<StatisticsNode>
        {
            public enum MaxMin { Max, Min, Median }
            [MyBrowsable, Category("Params"), Description("If more values have same frequency, return max/min of them")]
            [YAXSerializableField(DefaultValue = MaxMin.Max)]
            public MaxMin ChooseValue { get; set; }

            private int windowIdx { get; set; }
            private int elementsCountN { get; set; }
            private int columnsN, rowsN;
            private float[] m_actualConslVals;

            delegate float MaxMinFunction(IEnumerable<float> source);

            public override void Init(int nGPU)
            {
                windowIdx = 0;
                elementsCountN = 0;
                rowsN = Owner.Input.Count / Owner.Input.ColumnHint;
                columnsN = Owner.Input.ColumnHint;
                m_actualConslVals = new float[Owner.Output.Count];
            }

            public override void Execute()
            {
                MaxMinFunction mmf;
                if (ChooseValue == MaxMin.Max) mmf = Enumerable.Max;
                else if (ChooseValue == MaxMin.Min) mmf = Enumerable.Min;
                else if (ChooseValue == MaxMin.Median) mmf = Median;
                else { Debug.Assert(false, "Unknow parameter in StatisticsNode, ModeTask, ChooseValue"); return; }

                Owner.Input.SafeCopyToHost();
                Owner.Window.SafeCopyToHost();

                // calculate vals for actual matrix
                float[] m_values = Owner.Input.Host;
                switch (Owner.CalculateFor)
                {
                    case CalculateForEnum.AllElements:
                        m_actualConslVals = m_values;
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < columnsN; i++)
                        {
                            m_actualConslVals[i] = mmf(Mode(m_values, i, columnsN, rowsN));
                        }
                        break;
                    case CalculateForEnum.Rows:
                        for (int i = 0; i < rowsN; i++)
                        {
                            m_actualConslVals[i] = mmf(Mode(m_values, i * columnsN, 1, columnsN));
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        m_actualConslVals[0] = mmf(Mode(m_values));
                        break;
                }

                if (Owner.WindowLength == 1)
                {
                    m_actualConslVals.CopyTo(Owner.Window.Host, 0);
                    m_actualConslVals.CopyTo(Owner.Output.Host, 0);
                }
                else
                {
                    // calculate for whole window
                    if (elementsCountN < Owner.WindowLength) elementsCountN++;
                    for (int i = 0; i < Owner.Output.Count; i++, windowIdx++)
                    {
                        Owner.Window.Host[windowIdx] = m_actualConslVals[i];
                        Owner.Output.Host[i] = mmf(Mode(Owner.Window.Host, i, Owner.Output.Count, elementsCountN));
                    }
                }
                windowIdx %= Owner.Window.Count;

                Owner.Window.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }

            private List<float> Mode(float[] values, int startIdx, int step, int stepsN)
            {
                float[] selectedValues = new float[stepsN];

                for (int j = 0, i = startIdx; j < stepsN; j++, i += step)
                {
                    selectedValues[j] = values[i];
                }

                return Mode(selectedValues);
            }

            private List<float> Mode(float[] f)
            {
                var groups = f.GroupBy(v => v);
                int maxCount = groups.Max(g => g.Count());
                float mode = groups.First(g => g.Count() == maxCount).Key;
                var maxGroups = groups.Where(g => g.Count() == maxCount);
                List<float> r = new List<float>();
                foreach (var item in maxGroups) r.Add(item.Key);
                return r;
            }

            private float Median(IEnumerable<float> f)
            {
                int count = f.Count();
                var ordered = f.OrderBy(n => n);
                float r = (f.ElementAt(count/2) + f.ElementAt((count-1)/2)) / 2;
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
