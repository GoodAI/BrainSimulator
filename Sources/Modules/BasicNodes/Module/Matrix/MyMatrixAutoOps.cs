using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;


namespace GoodAI.Modules.Matrix
{
   

    /// <summary>
    /// Class for performing matrix operations
    /// </summary>
    /// 
    /// <h2> Usage </h2>
    ///   MyMatrixAutoOps mat_operation;
    /// 
    ///   //--- in init task
    ///   mat_operation = new MyMatrixAutoOps(Owner, Matrix.MatOperation.Multiplication | Matrix.MatOperation.Addition, A); // It may need A for setting up kernel size!  Operation is for 
    ///
    /// 
    ///  //--- when you need to use it
    ///  mat_operation.Run(Matrix.MatOperation.Multiplication, A, B, Output1);
    ///  mat_operation.Run(Matrix.MatOperation.Multiplication, A, 10.4f, Output2);
    ///  mat_operation.Run(Matrix.MatOperation.Addition, A, B, Output3);
    ///
    public class MyMatrixAutoOps : MyMatrixOps
    {
        private MyMatrixKernelOps MatKerlOps;
        private MyMatrixCublasOps MatCublOps;
        private MyMatrixCPUOps MatCPUOps;


        public MyMatrixAutoOps(MyWorkingNode callee, MatOperation operations, MyMemoryBlock<float> A = null)
        {
            if ((MyMatrixKernelOps.AvailableOperations() & operations) > 0)
            {
                MatKerlOps = new MyMatrixKernelOps(callee, operations);
            }
            if ((MyMatrixCublasOps.AvailableOperations() & operations) > 0)
            {
                MatCublOps = new MyMatrixCublasOps(callee);
            }
            if ((MyMatrixCPUOps.AvailableOperations() & operations) > 0)
            {
                MatCPUOps = new MyMatrixCPUOps(callee);
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            if ((MyMatrixCublasOps.AvailableOperations() & operation) > 0)
            {
                MatCublOps.Run(operation, A, B, Result);
            }
            else if ((MyMatrixKernelOps.AvailableOperations() & operation) > 0)
            {
                MatKerlOps.Run(operation, A, B, Result);
            }
            else if ((MyMatrixCPUOps.AvailableOperations() & operation) > 0)
            {
                MatCPUOps.Run(operation, A, B, Result);
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run undefined MatOps");
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, float value, MyMemoryBlock<float> Result)
        {
            if ((MyMatrixCublasOps.AvailableOperations() & operation) > 0)
            {
                MatCublOps.Run(operation, A, value, Result);
            }
            else if ((MyMatrixKernelOps.AvailableOperations() & operation) > 0)
            {
                MatKerlOps.Run(operation, A, value, Result);
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run undefined MatOps");
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result)
        {
            if ((MyMatrixCublasOps.AvailableOperations() & operation) > 0)
            {
                MatCublOps.Run(operation, A, Result);
            }
            else if ((MyMatrixKernelOps.AvailableOperations() & operation) > 0)
            {
                MatKerlOps.Run(operation, A, Result);
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run undefined MatOps");
            }
        }

        /// <summary>
        /// Enables overriding the dimensions to be applicable on tensors. Result columnHint seems not to be used for any operation so there is no need to override it.
        /// </summary>
        /// <param name="operation"></param>
        /// <param name="A"></param>
        /// <param name="Result"></param>
        /// <param name="AColumnHint">sees A as a matrix with this number of columns</param>
        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result, int AColumnHint)
        {
            if (A.Count % AColumnHint != 0)
            {
                throw new ArgumentException(string.Format("MyMatrixAutoOps: number of matrix elements ({0}) is not divisible by the desired column hint ({1})", A.Count, AColumnHint));
            }

            if ((MyMatrixCublasOps.AvailableOperations() & operation) > 0)
            {
                MatCublOps.Run(operation, A, Result, AColumnHint); 
            }
            else if ((MyMatrixKernelOps.AvailableOperations() & operation) > 0)
            {
                MatKerlOps.Run(operation, A, Result, AColumnHint);
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run undefined MatOps");
            }
        }

        public override void Run(MatOperation operation, MyMemoryBlock<float> A)
        {
            if ((MyMatrixCublasOps.AvailableOperations() & operation) > 0)
            {
                MatCublOps.Run(operation, A);
            }
            else if ((MyMatrixKernelOps.AvailableOperations() & operation) > 0)
            {
                MatKerlOps.Run(operation, A);
            }
            else
            {
                MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run undefined MatOps");
            }
        }


        public static MatOperation AvailableOperations()
        {
            return (MatOperation) (1 << Enum.GetValues(typeof(MatOperation)).Length); // it is integer of binary repre of the last one which is one moved n-times :)
        }
    }








}
