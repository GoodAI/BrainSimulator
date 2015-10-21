using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Schema;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using YAXLib;

namespace GoodAI.Modules.Join
{
    /// <summary>
    /// 
    /// </summary>
    [Flags]
    public enum MyStackingOperation
    {
        None = 0,

        Concatenate = 1 << 1,
        Interweave = 1 << 2,
    }

    /*** complete list of operations that should be included and IMPLEMENTED! ***/
#if false
    public enum MatOperation
    {
        Concatenate,
        Interweave,
    }
#endif


    public class MyStackingOps
    {
        private readonly MyWorkingNode _caller;
        private readonly MyStackingOperation _operations;
        private bool _forceInputChecking;

        public MyStackingOps(MyWorkingNode caller, MyStackingOperation operations, bool forceInputChecking = false)
        {
            _caller = caller;
            _operations = operations;
            _forceInputChecking = forceInputChecking;
        }



        public void Run(MyStackingOperation operation, MyMemoryBlock<float> output, params MyMemoryBlock<float>[] inputs)
        {
            if (inputs == null)
                MyLog.WARNING.WriteLine("No inputs for stacking operation to run. Owner: " + _caller.Name);
            else
                Run(operation, output, inputs.AsEnumerable());
        }

        public void Run(MyStackingOperation operation, MyMemoryBlock<float> output, IEnumerable<MyMemoryBlock<float>> inputs)
        {
            if (inputs == null || !inputs.Any())
            {
                MyLog.WARNING.WriteLine("No inputs for stacking operation to run. Owner: " + _caller.Name);
                return;
            }

            inputs = inputs.Where(a => a != null);

            if (_forceInputChecking && !Validate(operation, inputs, output))
                return;


            switch (operation)
            {
                case MyStackingOperation.Concatenate:
                    {
                        int resPtr = 0;

                        foreach (var input in inputs)
                        {
                            output.CopyFromMemoryBlock(input, 0, resPtr, input.Count);
                            resPtr += input.Count;
                        }
                    }
                    break;

                case MyStackingOperation.Interweave:
                    {
                        int i = 0;
                        int resPtr = 0;
                        var first = inputs.First();
                        int rows = first.Count / first.ColumnHint;


                        // Alter between copying the inputs, up to the last row
                        for (; i < rows - 1; i++)
                        {
                            foreach (var input in inputs)
                            {
                                output.CopyFromMemoryBlock(input, i * input.ColumnHint, resPtr, input.ColumnHint);
                                resPtr += input.ColumnHint;
                            }
                        }


                        // Copy the last rows
                        foreach (var input in inputs)
                        {
                            int lastRowIdx = i * input.ColumnHint;
                            int lastRowSize = input.Count - lastRowIdx;

                            if (lastRowSize > 0)
                            {
                                output.CopyFromMemoryBlock(input, lastRowIdx, resPtr, lastRowSize);
                                resPtr += lastRowSize;
                            }
                        }
                    }
                    break;
            }
        }


        public bool Validate(MyStackingOperation operation, IEnumerable<MyMemoryBlock<float>> inputs, MyMemoryBlock<float> output)
        {
            if (operation == MyStackingOperation.None)
                return false;

            if ((operation & _operations) == 0)
            {
                MyLog.WARNING.WriteLine("Trying to execute an uninitialized stacking operation. Owner: " + _caller.Name);
                return false;
            }

            return true;
        }

        public static bool Validate(MyStackingOperation operation, IEnumerable<MyMemoryBlock<float>> inputs, MyMemoryBlock<float> output, out string errorOutput)
        {
            errorOutput = null;

            inputs = inputs.Where(a => a != null);

            switch (operation)
            {
                case MyStackingOperation.None:
                    return true;

                case MyStackingOperation.Concatenate:
                case MyStackingOperation.Interweave:
                    break;

                default:
                    errorOutput = "Invalid operation. Only a single value within the enum range should be passed.";
                    return false;
            }

            if (!inputs.Any())
            {
                errorOutput = "No inputs for stacking operation to run.";
                return false;
            }


            if (operation == MyStackingOperation.Interweave)
            {
                if (inputs.Any(a => a.ColumnHint == 0))
                {
                    errorOutput = "Invalid column hints. They must be non-negative.";
                    return false;
                }

                var first = inputs.First();
                int rows = first.Count / first.ColumnHint;

                if (inputs.Any(a => a.Count / a.ColumnHint != rows))
                {
                    errorOutput = "Invalid input row counts. Inputs must have the same number of rows.";
                    return false;
                }
            }

            if (inputs.Sum(a => a.Count) > output.Count)
            {
                errorOutput = "Invalid output size: " + output.Count + ". Must be large enough to contain all the inputs.";
                return false;
            }


            return true;
        }
    }
}
