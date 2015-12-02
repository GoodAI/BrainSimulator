using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Common
{
    /// <author>GoodAI</author>
    /// <meta>ms</meta>
    /// <status>Working</status>
    /// <summary> For given data returns 5-tuple (Min, FirstQuartile, Median, ThirdQuartile, Max). </summary>
    /// <description>
    /// <p>5-tuples are in rows (ColumnHint = 5). </p>
    /// There are four modes of input processing: 
    /// <ul> 
    /// <li>Take data from whole input matrix: CalculateFor = <b>WholeMatrix</b></li>
    /// <li>Take data from rows: CalculateFor = <b>Rows</b></li>
    /// <li>Take data from columns: CalculateFor = <b>Columns</b></li>
    /// <li>Take data for every element from time-collecting window: CalculateFor = <b>AllElements</b>; WindowLength = <b>X</b> >= 3</li>
    /// </ul>
    /// </description>
    public class BoxPlotNode : MyWorkingNode
    {
        #region inputs

        [MyBrowsable, Category("History")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("History")]
        public int WindowLength { get; set; }

        public enum CalculateForEnum { AllElements, Rows, Columns, WholeMatrix }
        [MyBrowsable, Category("\tConsolidation"), Description(" ")]
        [YAXSerializableField(DefaultValue = CalculateForEnum.WholeMatrix)]
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

        public BoxPlotTask BoxPlot { get; private set; }

        public int OutputRowsN { get; private set; }
        public int OutputColumnsN { get; private set; }
        public override void UpdateMemoryBlocks()
        {
            if (Input == null) return;

            int inputSize = Input == null ? 1 : Input.Count;
            int inputRowN = inputSize / Input.ColumnHint;
            int inputColumnN = Input.ColumnHint;

            if (CalculateFor == CalculateForEnum.AllElements) { Output.Count = 5 * inputSize; Temp.Count = WindowLength; }
            else if (CalculateFor == CalculateForEnum.Rows) { Output.Count = 5 * inputColumnN; Temp.Count = inputRowN; }
            else if (CalculateFor == CalculateForEnum.Columns) { Output.Count = inputRowN * 5; Temp.Count = inputColumnN; }
            else if (CalculateFor == CalculateForEnum.WholeMatrix) { Output.Count = 5; Temp.Count = inputSize; }
            Window.Count = inputSize * WindowLength;
            Output.ColumnHint = 5;
            OutputRowsN = Output.Count / Output.ColumnHint;
            OutputColumnsN = Output.ColumnHint;
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(!(BoxPlot.Enabled && WindowLength != 1 && CalculateFor != CalculateForEnum.AllElements), this,
                "BoxPlot with WindowLength other than 1 can be used with CalculateFor = 'AllElements' option only");
            validator.AssertError(!(BoxPlot.Enabled && CalculateFor == CalculateForEnum.WholeMatrix && Input.Count < 5), this,
                "BoxPlot recieves too small input to get reasonable output");
        }

        public override string Description
        {
            get
            {
                if (BoxPlot.Enabled) { return BoxPlot.Description; }
                else return base.Description;
            }
        }

        /// <summary>Returns 5-tuple (Min, FirstQuartile, Median, ThirdQuartile, Max).</summary>
        [Description("BoxPlot")]
        public class BoxPlotTask : MyTask<BoxPlotNode>
        {
            private int windowIdx { get; set; }
            private int elementsCount { get; set; }
            private int columnsN, rowsN, elementsN, outputRowsN;
            MyCudaKernel m_sumKernel;
            private readonly int HOST_DEVICE_THRSHD = 1 << 13;

            private bool firstExec;

            public override void Init(int nGPU)
            {
                firstExec = true;

                windowIdx = 0;
                elementsCount = 0;
                elementsN = Owner.Input.Count;
                columnsN = Owner.Input.ColumnHint;
                rowsN = elementsN / columnsN;
                //m_sumKernel = MyReductionFactory.Kernel(nGPU, MyReductionFactory.Mode.f_Sum_f);
                Owner.Window.SafeCopyToHost();
            }

            public override void Execute()
            {
                if (firstExec) {
                    firstExec = false;
                    for (int i = 0; i < Owner.Window.Count; i++)
                    {
                        Owner.Window.Host[i] = float.NegativeInfinity;
                    }
                    Owner.Window.SafeCopyToDevice();
                }
                Owner.Input.SafeCopyToHost();

                switch (Owner.CalculateFor)
                {
                    case CalculateForEnum.AllElements:
                        for (int i = 0; i < elementsN; i++)
                        {
                            int WindowStartIdx = i * Owner.WindowLength;
                            Owner.Window.Host[WindowStartIdx + windowIdx] = Owner.Input.Host[i];
                            // Owner.Window.SafeCopyToDevice(); // optimize!

                            // activate for GPU-sorting
                            //Owner.Window.CopyToMemoryBlock(Owner.Temp, WindowStartIdx, 0, Owner.WindowLength);
                            // deactivate for GPU-sorting

                            // optimize memory handling before GPU-sort activating !!!
                            Array.Copy(Owner.Window.Host, WindowStartIdx, Owner.Temp.Host, 0, Owner.WindowLength);

                            Sort(Owner.Temp);
                            if (elementsCount < 3) continue;
                            inputAssignment(i, Owner.Temp.Host, elementsCount);
                        }
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < rowsN; i++)
                        {
                            Array.Copy(Owner.Input.Host, i * columnsN, Owner.Temp.Host, 0, columnsN);

                            Sort(Owner.Temp);
                            inputAssignment(i, Owner.Temp.Host);
                        }
                        break;

                    case CalculateForEnum.Rows:
                        for (int i = 0; i < columnsN; i++)
                        {
                            for (int j = 0; j < rowsN; j++)
                            {
                                Owner.Temp.Host[j] = Owner.Input.Host[i * rowsN + j];
                            }
                            Sort(Owner.Temp);
                            inputAssignment(i, Owner.Temp.Host);
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        Array.Copy(Owner.Input.Host, Owner.Temp.Host, elementsN);

                        Sort(Owner.Temp);
                        inputAssignment(0, Owner.Temp.Host);
                        break;
                }
                windowIdx++;
                windowIdx %= Owner.WindowLength;
                if (elementsCount < Owner.WindowLength - 1) elementsCount++;

                Owner.Window.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }

            private void inputAssignment(int idx, float[] sortedInput)
            {
                Owner.Output.Host[idx * 5 + 0] = getMin(sortedInput);
                Owner.Output.Host[idx * 5 + 1] = getFirstQuartile(sortedInput);
                Owner.Output.Host[idx * 5 + 2] = getMedian(sortedInput);
                Owner.Output.Host[idx * 5 + 3] = getThirdQuartile(sortedInput);
                Owner.Output.Host[idx * 5 + 4] = getMax(sortedInput);
            }

            // for base < SortedInput.lenght()
            private void inputAssignment(int idx, float[] sortedInput, int elementsCount)
            {
                int MinIdx = sortedInput.Length - elementsCount - 1;
                int MaxIdx = sortedInput.Length - 1;
                if (elementsCount == 0) return;
                else if (elementsCount == 1)
                {
                    for (int i = 0; i < 5; i++ )
                    {
                        Owner.Output.Host[idx * 5 + i] = sortedInput[0];
                    }
                    return;
                }
                
                Owner.Output.Host[idx * 5 + 0] = sortedInput[MinIdx];
                Owner.Output.Host[idx * 5 + 1] = getFirstQuartile(sortedInput, MinIdx, MaxIdx);
                Owner.Output.Host[idx * 5 + 2] = getMedian(sortedInput, MinIdx, MaxIdx);
                Owner.Output.Host[idx * 5 + 3] = getThirdQuartile(sortedInput, MinIdx, MaxIdx);
                Owner.Output.Host[idx * 5 + 4] = getMax(sortedInput);
            }

            private float getFirstQuartile(float[] sortedInput)
            {
                if (sortedInput.Length % 2 == 1)
                {
                    return getMedian(sortedInput, 0, sortedInput.Length / 2);
                }
                return getMedian(sortedInput, 0, sortedInput.Length / 2 - 1);
            }

            private float getFirstQuartile(float[] sortedInput, int minIdx, int maxIdx)
            {
                int length = (maxIdx - minIdx);
                if (length % 2 == 1)
                {
                    return getMedian(sortedInput, minIdx, minIdx + length / 2);
                }
                return getMedian(sortedInput, minIdx, minIdx + length / 2 - 1);
            }

            private float getThirdQuartile(float[] sortedInput)
            {
                if (sortedInput.Length % 2 == 1)
                {
                    return getMedian(sortedInput, sortedInput.Length / 2, sortedInput.Length - 1);
                }
                return getMedian(sortedInput, sortedInput.Length / 2 - 1, sortedInput.Length - 1);
            }

            private float getThirdQuartile(float[] sortedInput, int minIdx, int maxIdx)
            {
                int length = (maxIdx - minIdx);
                return getMedian(sortedInput, minIdx + length / 2, minIdx + length - 1);
            }

            private float getMedian(float[] sortedInput)
            {
                return getMedian(sortedInput, 0, sortedInput.Length - 1);
            }

            private float getMedian(float[] sortedInput, int minIdx, int maxIdx)
            {
                int length = maxIdx - minIdx + 1;

                if (length == 0) return float.NaN;
                if (length == 1) return sortedInput[minIdx];

                if (length % 2 == 1)
                {
                    int MidIdx = minIdx + length / 2;
                    return sortedInput[MidIdx];
                }
                int MidIdx0 = minIdx + length / 2;
                int MidIdx1 = MidIdx0 - 1;
                return (sortedInput[MidIdx0] + sortedInput[MidIdx1]) / 2;
            }

            private float getMin(float[] sortedOutput)
            {
                return sortedOutput[0];
            }

            private float getMax(float[] sortedOutput)
            {
                return sortedOutput[sortedOutput.Length - 1];
            }

            private void Sort(MyMemoryBlock<float> values)
            {
                Array.Sort(values.Host);
                // use THRUST library insead
            }


            public string Description
            {
                get
                {
                    switch (Owner.CalculateFor)
                    {
                        case CalculateForEnum.AllElements: return "BoxPlot of elements";
                        case CalculateForEnum.Rows: return "BoxPlot of rows";
                        case CalculateForEnum.Columns: return "BoxPlot of columns";
                        case CalculateForEnum.WholeMatrix: return "BoxPlot";
                        default: MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Using BoxPlotTask with unknown parameter"); return "";
                    }
                }
            }
        }
    }
}
