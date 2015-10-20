using GoodAI.Core;           // manual kernel sizes are needed
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System.Collections.Generic;


namespace GoodAI.Modules.Matrix
{



    /// <summary>
    ///    Perform operations that are defined by kernel
    /// </summary>
    public class MyMatrixKernelOps : MyMatrixOps
    {
        private Dictionary<MatOperation, MyCudaKernel> OpersKerlsDictionary;

        public MyMatrixKernelOps(MyWorkingNode callee, MatOperation operations, MyMemoryBlock<float> A = null, MyMemoryBlock<float> B = null)
        {
            OpersKerlsDictionary = new Dictionary<MatOperation, MyCudaKernel>();
            this.callee = callee;

            if ((operations & MatOperation.Log) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Log, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "LogKernel_naive"));
            }
            if ((operations & MatOperation.Exp) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Exp, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "ExpKernel_naive"));
            }
            if ((operations & MatOperation.Round) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Round, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "RoundKernel_naive"));
            }
            if ((operations & MatOperation.Floor) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Floor, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "FloorKernel_naive"));
            }
            if ((operations & MatOperation.Ceil) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Ceil, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "CeilKernel_naive"));
            }
            if ((operations & MatOperation.Abs) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Abs, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "AbsKernel_naive"));
            }
            if ((operations & MatOperation.GetCol) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.GetCol, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_getCol_FloatId_naive"));
            }
            if ((operations & MatOperation.GetRow) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.GetRow, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_getRow_FloatId_naive"));
            }
            if ((operations & MatOperation.MultiplElemntWise) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.MultiplElemntWise, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_MultiplElementWise_naive"));
            }
            if ((operations & MatOperation.Addition) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Addition, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_Addition_naive"));
            }
            if ((operations & MatOperation.Pow) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Pow, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_Pow_naive"));
            }
            if ((operations & MatOperation.Substraction) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Substraction, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_Substraction_naive"));
            }
            if ((operations & MatOperation.Transpose) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.Transpose, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_transposeFromSVDnodeCOPY"));
            }
            if ((operations & MatOperation.PermuteRows) > 0)
            {
                OpersKerlsDictionary.Add(MatOperation.PermuteRows, MyKernelFactory.Instance.Kernel(callee.GPU, @"Vision\Matrix", "Matrix_PermuteRows"));
            }
            if (operations > 0 && OpersKerlsDictionary.Count == 0)
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to init kernel MatrixOps for undefined MatOperation");
            }
        }

        //Round a / b to nearest higher integer value
        private int iDivUp(int a, int b)
        {
            return (a % b != 0) ? (a / b + 1) : (a / b);
        }

        //Round a / b to nearest lower integer value
        private int iDivDown(int a, int b)
        {
            return a / b;
        }

        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            if (OpersKerlsDictionary.ContainsKey(operation))
            {
                if (operation == MatOperation.GetRow)
                {
                    B.SafeCopyToHost();
                    OpersKerlsDictionary[operation].SetupExecution(A.ColumnHint);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint, B.Host[0]);
                }
                else if (operation == MatOperation.GetCol)
                {
                    B.SafeCopyToHost();
                    OpersKerlsDictionary[operation].SetupExecution(A.Count / A.ColumnHint);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint, B.Host[0]);
                }
                else if (operation == MatOperation.MultiplElemntWise | operation == MatOperation.Addition | operation == MatOperation.Substraction | operation == MatOperation.Pow)
                {
                    if (A.Count >= B.Count)
                    {
                        OpersKerlsDictionary[operation].SetupExecution(A.Count);
                        OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, B, B.Count, B.ColumnHint, Result, Result.Count, Result.ColumnHint, float.NaN);
                    }
                    else
                    {
                        OpersKerlsDictionary[operation].SetupExecution(B.Count);
                        OpersKerlsDictionary[operation].Run(B, B.Count, B.ColumnHint, A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint, float.NaN);
                    }
                }
                else
                { // other executions are performed by ,,standartezied'' kernel-call
                    OpersKerlsDictionary[operation].SetupExecution(A.Count);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, B, B.Count, B.ColumnHint, Result, Result.Count, Result.ColumnHint);
                }
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run kernel MatrixOps for uninitialized kernel");
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A)
        {

        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result)
        {
            if (OpersKerlsDictionary.ContainsKey(operation))
            {
                OpersKerlsDictionary[operation].SetupExecution(A.Count);
                OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint);
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, float value, MyMemoryBlock<float> Result)
        {
            if (OpersKerlsDictionary.ContainsKey(operation))
            {
                if (operation == MatOperation.GetRow)
                {
                    OpersKerlsDictionary[operation].SetupExecution(A.ColumnHint);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint, value);
                }
                if (operation == MatOperation.GetCol)
                {
                    OpersKerlsDictionary[operation].SetupExecution(A.Count / A.ColumnHint);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint, value);
                }
                else if (operation == MatOperation.MultiplElemntWise | operation == MatOperation.Addition | operation == MatOperation.Pow)
                {
                    OpersKerlsDictionary[operation].SetupExecution(A.Count);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, A, 0, 0, Result, Result.Count, Result.ColumnHint, value);
                }
                else
                {
                    OpersKerlsDictionary[operation].SetupExecution(A.Count);
                    OpersKerlsDictionary[operation].Run(A, A.Count, A.ColumnHint, Result, Result.Count, Result.ColumnHint, value);
                }
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run kernel MatrixOps for uninitialized kernel");
            }
        }




        public static MatOperation AvailableOperations()
        {
            return MatOperation.GetRow | MatOperation.GetCol | MatOperation.Exp | MatOperation.MultiplElemntWise | MatOperation.Addition | MatOperation.Log | MatOperation.Pow | MatOperation.Exp | MatOperation.Round | MatOperation.Floor | MatOperation.Ceil | MatOperation.Abs | MatOperation.Substraction | MatOperation.Transpose | MatOperation.PermuteRows;
        }



    }


}
