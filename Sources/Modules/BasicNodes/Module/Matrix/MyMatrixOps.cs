using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using System;

namespace GoodAI.Modules.Matrix
{
    /// <summary>
    /// Operations that are allowed to run using the Matrix Node
    /// </summary>
    [Flags]
    public enum MatOperation
    {
        None = 0,
        Addition = 1,
        Multiplication = 1 << 2,

        DotProd = 1 << 3,
        MultiplElemntWise = 1 << 4,
        Substraction = 1 << 5,

        AbsMinIndex = 1 << 6,
        AbsMaxIndex = 1 << 7,

        GetCol = 1 << 8,
        GetRow = 1 << 9,

        Minus = 1 << 10,


        Normalize = 1 << 11,
        Norm2 = 1 << 12,


        EuclidDist = 1 << 13,

        Pow = 1 << 14,
        Exp = 1 << 15,
        Log = 1 << 16,

        Abs = 1 << 17,
        Floor = 1 << 18,
        Round = 1 << 19,
        Ceil = 1 << 20,

        Copy = 1 << 21,

        Transpose = 1 << 22,
        
        PermuteRows = 1 << 23
    }



    /// <summary>
    /// Strategy DesignPatern:
    ///    This is the abstract class that defines what will happen, then specific instance (that depends on the execuion=operation type (CPU/GPU/cublas..)) will execute the queried operation
    /// </summary>
    public abstract class MyMatrixOps
    {
        protected MyWorkingNode callee;


        public abstract void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result);
        public abstract void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result);
        public abstract void Run(MatOperation operation, MyMemoryBlock<float> A, float value, MyMemoryBlock<float> Result);
        public abstract void Run(MatOperation operation, MyMemoryBlock<float> A); // change A


        public static MatOperation AvailableOperations()
        {
            return (MatOperation)0;
        }





        public float RunReturn(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            Run(operation, A, B, Result);
            Result.SafeCopyToHost();
            return Result.Host[0];
        }
        public float RunReturn(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result)
        {
            Run(operation, A, Result);
            Result.SafeCopyToHost();
            return Result.Host[0];
        }



        public static MyMemoryBlock<float> SetupResultSize(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            Result.Count = A != null ? A.Count : 1;
            Result.ColumnHint = A != null ? A.ColumnHint : 1;

            if (A != null)
            {
                if (operation == MatOperation.DotProd)
                {
                    Result.Count = Result.ColumnHint = 1;
                }
                else if (operation == MatOperation.AbsMinIndex || operation == MatOperation.AbsMaxIndex)
                {
                    Result.ColumnHint = 1;
                    Result.Count = 1;
                }
                else if (operation == MatOperation.Multiplication)
                {
                    if (A != null && B != null && A.ColumnHint != 0 && B.Count > 1)
                    {
                        Result.ColumnHint = B.ColumnHint;
                        Result.Count = B.ColumnHint * A.Count / A.ColumnHint;
                    }
                }
                else if (operation == MatOperation.GetCol)
                {
                    Result.Count = A.Count / A.ColumnHint;
                    Result.ColumnHint = Result.Count;
                }
                else if (operation == MatOperation.GetRow)
                {
                    Result.Count = A.ColumnHint;
                    Result.ColumnHint = Result.Count;
                }
                else if (B != null && (operation == MatOperation.MultiplElemntWise || operation == MatOperation.Addition))
                {
                    Result.ColumnHint = Math.Max(A.ColumnHint, B.ColumnHint);
                    Result.Count = Math.Max(A.Count, B.Count);
                }
                else if (operation == MatOperation.Transpose)
                {
                    Result.Dims = A.Dims.Transpose();
                }
                else if (operation == MatOperation.EuclidDist)
                {
                    if (B != null)
                    {
                        Result.Count = A.Count / A.ColumnHint;
                        Result.ColumnHint = 1;
                    }

                }
            }
            return Result;
        }


        public static bool Validate(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            if (A == null || Result == null)
                return false;
            bool is_it_correct = true;

            if (operation == MatOperation.DotProd)
            {
                is_it_correct = (A.Count == B.Count) && (Result.Count == 1);
            }
            else if (operation == MatOperation.Multiplication)
            { // it should allow MAT*MAT, vec*MAT and MAT*vec , in correct sizes of course
                if (B == null)
                {
                    is_it_correct = A.Count == Result.Count && A.ColumnHint == Result.ColumnHint;
                }
                else
                {
                    is_it_correct = (A.ColumnHint == B.Count / B.ColumnHint) && (B.ColumnHint == Result.ColumnHint) && (A.Count / A.ColumnHint == Result.Count / Result.ColumnHint);
                    is_it_correct = is_it_correct || (B.Count == 1) || (A.Count == 1); // it still allows A*5 :-)
                }
            }
            else if (operation == MatOperation.Addition || operation == MatOperation.MultiplElemntWise)
            {
                if (B == null)
                {
                    is_it_correct = A.Count == Result.Count && A.ColumnHint == Result.ColumnHint;
                }
                else
                {
                    is_it_correct = (A.Count == B.Count) && (A.ColumnHint == B.ColumnHint);  // same size
                    is_it_correct |= A.ColumnHint == B.ColumnHint || A.Count / A.ColumnHint == B.Count / B.ColumnHint; // same # of colums, rows
                    is_it_correct |= A.Count == 1 || B.Count == 1;
                    is_it_correct |= (Math.Max(A.Count, B.Count) == Result.Count) && (Math.Max(A.ColumnHint, B.ColumnHint) == Result.ColumnHint);
                }
            }
            else if (operation == MatOperation.EuclidDist)
            {
                is_it_correct = (A.ColumnHint == B.Count);
            }
            return is_it_correct;

        }




    }
}
