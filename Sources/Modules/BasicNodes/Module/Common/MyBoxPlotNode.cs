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
    public class MyBoxPlotNode : MyWorkingNode
    {
        #region inputs
        [MyBrowsable, Category("Input interface")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int INPUT_SIZE { get; private set; }

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

            INPUT_SIZE = Input == null ? 1 : Input.Count;
            int InputRowN = INPUT_SIZE / Input.ColumnHint;
            int InputColumnN = Input.ColumnHint;

            if (CalculateFor == CalculateForEnum.AllElements) { Output.Count = 5 * INPUT_SIZE; Temp.Count = WindowLength; }
            else if (CalculateFor == CalculateForEnum.Rows) { Output.Count = 5 * InputColumnN; Temp.Count = InputRowN; }
            else if (CalculateFor == CalculateForEnum.Columns) { Output.Count = InputRowN * 5; Temp.Count = InputColumnN; }
            else if (CalculateFor == CalculateForEnum.WholeMatrix) { Output.Count = 5; Temp.Count = INPUT_SIZE; }
            Window.Count = INPUT_SIZE * WindowLength;
            Output.ColumnHint = 5;
            OutputRowsN = Output.Count / Output.ColumnHint;
            OutputColumnsN = Output.ColumnHint;
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(!(BoxPlot.Enabled && WindowLength != 1 && CalculateFor != CalculateForEnum.AllElements), this,
                "BoxPlot with WindowLength other than 1 can be used with CalculateFor = 'AllElements' option only");
            validator.AssertError(!(BoxPlot.Enabled && CalculateFor == CalculateForEnum.WholeMatrix && INPUT_SIZE < 5), this,
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
        public class BoxPlotTask : MyTask<MyBoxPlotNode>
        {
            private int WindowIdx { get; set; }
            private int Base { get; set; }
            private int ColumnsN, RowsN, ElementsN, OutputRowsN;
            MyCudaKernel m_sumKernel;
            private readonly int HOST_DEVICE_THRSHD = 1 << 13;

            private bool firstExec;
            public override void Init(int nGPU)
            {
                firstExec = true;

                WindowIdx = 0;
                Base = 0;
                ColumnsN = Owner.Input.ColumnHint;
                RowsN = Owner.INPUT_SIZE / Owner.Input.ColumnHint;
                ColumnsN = Owner.Input.ColumnHint;
                ElementsN = Owner.INPUT_SIZE;
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
                        for (int i = 0; i < ElementsN; i++)
                        {
                            int WindowStartIdx = i * Owner.WindowLength;
                            Owner.Window.Host[WindowStartIdx + WindowIdx] = Owner.Input.Host[i];
                            // Owner.Window.SafeCopyToDevice(); // optimize!

                            // activate for GPU-sorting
                            //Owner.Window.CopyToMemoryBlock(Owner.Temp, WindowStartIdx, 0, Owner.WindowLength);
                            // deactivate for GPU-sorting

                            // optimize memory handling before GPU-sort activating !!!
                            Array.Copy(Owner.Window.Host, WindowStartIdx, Owner.Temp.Host, 0, Owner.WindowLength);

                            Sort(Owner.Temp);
                            if (Base < 3) continue;
                            inputAssignment(i, Owner.Temp.Host, Base);
                        }
                        break;
                    case CalculateForEnum.Columns:
                        for (int i = 0; i < RowsN; i++)
                        {
                            Array.Copy(Owner.Input.Host, i * ColumnsN, Owner.Temp.Host, 0, ColumnsN);

                            Sort(Owner.Temp);
                            inputAssignment(i, Owner.Temp.Host);
                        }
                        break;

                    case CalculateForEnum.Rows:
                        for (int i = 0; i < ColumnsN; i++)
                        {
                            for (int j = 0; j < RowsN; j++)
                            {
                                Owner.Temp.Host[j] = Owner.Input.Host[i * RowsN + j];
                            }
                            Sort(Owner.Temp);
                            inputAssignment(i, Owner.Temp.Host);
                        }
                        break;
                    case CalculateForEnum.WholeMatrix:
                        Array.Copy(Owner.Input.Host, Owner.Temp.Host, ElementsN);

                        Sort(Owner.Temp);
                        inputAssignment(0, Owner.Temp.Host);
                        break;
                }
                WindowIdx++;
                WindowIdx %= Owner.WindowLength;
                if (Base < Owner.WindowLength - 1) Base++;

                Owner.Window.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }

            private void inputAssignment(int Idx, float[] SortedInput)
            {
                Owner.Output.Host[Idx * 5 + 0] = getMin(SortedInput);
                Owner.Output.Host[Idx * 5 + 1] = getFirstQuartile(SortedInput);
                Owner.Output.Host[Idx * 5 + 2] = getMedian(SortedInput);
                Owner.Output.Host[Idx * 5 + 3] = getThirdQuartile(SortedInput);
                Owner.Output.Host[Idx * 5 + 4] = getMax(SortedInput);
            }

            // for base < SortedInput.lenght()
            private void inputAssignment(int Idx, float[] SortedInput, int Base)
            {
                int MinIdx = SortedInput.Length - Base - 1;
                int MaxIdx = SortedInput.Length - 1;
                if (Base == 0) return;
                else if (Base == 1)
                {
                    for (int i = 0; i < 5; i++ )
                    {
                        Owner.Output.Host[Idx * 5 + i] = SortedInput[0];
                    }
                    return;
                }
                
                Owner.Output.Host[Idx * 5 + 0] = SortedInput[MinIdx];
                Owner.Output.Host[Idx * 5 + 1] = getFirstQuartile(SortedInput, MinIdx, MaxIdx);
                Owner.Output.Host[Idx * 5 + 2] = getMedian(SortedInput, MinIdx, MaxIdx);
                Owner.Output.Host[Idx * 5 + 3] = getThirdQuartile(SortedInput, MinIdx, MaxIdx);
                Owner.Output.Host[Idx * 5 + 4] = getMax(SortedInput);
            }

            private float getFirstQuartile(float[] SortedInput)
            {
                if (SortedInput.Length % 2 == 1)
                {
                    return getMedian(SortedInput, 0, SortedInput.Length / 2);
                }
                return getMedian(SortedInput, 0, SortedInput.Length / 2 - 1);
            }

            private float getFirstQuartile(float[] SortedInput, int MinIdx, int MaxIdx)
            {
                int length = (MaxIdx - MinIdx);
                if (length % 2 == 1)
                {
                    return getMedian(SortedInput, MinIdx, MinIdx + length / 2);
                }
                return getMedian(SortedInput, MinIdx, MinIdx + length / 2 - 1);
            }

            private float getThirdQuartile(float[] SortedInput)
            {
                if (SortedInput.Length % 2 == 1)
                {
                    return getMedian(SortedInput, SortedInput.Length / 2, SortedInput.Length - 1);
                }
                return getMedian(SortedInput, SortedInput.Length / 2 - 1, SortedInput.Length - 1);
            }

            private float getThirdQuartile(float[] SortedInput, int MinIdx, int MaxIdx)
            {
                int length = (MaxIdx - MinIdx);
                return getMedian(SortedInput, MinIdx + length / 2, MinIdx + length - 1);
            }

            private float getMedian(float[] SortedInput)
            {
                return getMedian(SortedInput, 0, SortedInput.Length - 1);
            }

            private float getMedian(float[] SortedInput, int MinIdx, int MaxIdx)
            {
                int Length = MaxIdx - MinIdx + 1;

                if (Length == 0) return float.NaN;
                if (Length == 1) return SortedInput[MinIdx];

                if (Length % 2 == 1)
                {
                    int MidIdx = MinIdx + Length / 2;
                    return SortedInput[MidIdx];
                }
                int MidIdx0 = MinIdx + Length / 2;
                int MidIdx1 = MidIdx0 - 1;
                return (SortedInput[MidIdx0] + SortedInput[MidIdx1]) / 2;
            }

            private float getMin(float[] SortedOutput)
            {
                return SortedOutput[0];
            }

            private float getMax(float[] SortedOutput)
            {
                return SortedOutput[SortedOutput.Length - 1];
            }

            private void Sort(MyMemoryBlock<float> MMB)
            {
                Array.Sort(MMB.Host);
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
