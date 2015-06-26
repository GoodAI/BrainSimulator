using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Transforms;
using BrainSimulator.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

using BrainSimulator.Matrix;



namespace BrainSimulator.Matrix
{

    public class MyCublasFactory// : IDisposable
    {
        private static CudaBlas SINGLETON;

        public static CudaBlas Instance
        {
            get
            {
                if (SINGLETON == null)
                {
                    SINGLETON = new CudaBlas();
                }
                return SINGLETON;
            }
        }
    }



    /// <summary>
    /// 
    /// </summary>
    public class MyMatrixCublasOps : MyMatrixOps
    {
        //private CudaBlas cublas = null;


        public MyMatrixCublasOps(MyWorkingNode callee, MatOperation operation = 0, MyMemoryBlock<float> A = null, MyMemoryBlock<float> tmp = null)
        {
          //  cublas = new CudaBlas();
            this.callee = callee;
        }





        public void Run(MatOperation operation, CudaDeviceVariable<float> A, int ACount, int AColumnHint, CudaDeviceVariable<float> B, int BCount, int BColumnHint, CudaDeviceVariable<float> Result, int ResultCount, int ResultColumnHint)
        {
            Result.Memset(BitConverter.ToUInt32(BitConverter.GetBytes(0.0f), 0));
                
            switch (operation)
            {
                case MatOperation.Multiplication:  // vectors/matrices have to be always in the correct dimesions!
                    if (BCount > 1 && ACount > 1 && BColumnHint == 1 && ACount / AColumnHint > 1 && BCount / BColumnHint == AColumnHint) //. A*vecB
                    {
                        MyCublasFactory.Instance.Gemv(Operation.Transpose,  // transpose beacuase it does Ax row wise if x is a row vector :D
                            AColumnHint, ACount / AColumnHint, 1.0f,
                            A, AColumnHint,
                            B, 1,
                            1.0f, Result, 1);
                    }
                    else if (ACount > 1 && BCount > 1 && ACount / AColumnHint == 1 && BColumnHint > 1 && BCount / BColumnHint == AColumnHint)  // vecA*B
                    {
                        MyCublasFactory.Instance.Gemv(Operation.NonTranspose,  // transpose beacuase it does Ax row wise if x is a row vector :D
                            BColumnHint, BCount / BColumnHint, 1.0f,
                            B, BColumnHint,
                            A, 1,
                            1.0f, Result, 1);
                    }
                    else if (ACount / AColumnHint == 1 && BColumnHint == 1 && ACount > 1 && BCount > 1) //. trans(vecA) * vecB
                    {
                        Run(MatOperation.DotProd, A, ACount, AColumnHint, B, BCount, BColumnHint, Result, ResultCount, ResultColumnHint);
                    }
                    else if(ACount != 1 || BCount != 1)// A*B   matrix multiplication
                    {
                        MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                            ACount / AColumnHint, BColumnHint, AColumnHint, 1.0f,
                            A, ACount / AColumnHint,
                            B, BCount / BColumnHint,
                            1.0f, Result, ResultColumnHint);
                    }
                    break;
                case MatOperation.DotProd:
                    MyCublasFactory.Instance.Gemv(Operation.Transpose,  // transpose beacuase it does Ax row wise if x is a row vector :D
                       ACount, 1, 1.0f,
                       A, ACount,
                       B, 1,
                       1.0f, Result, 1);
                    break;
                default:
                    MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cublas for undefined MatOperation");
                    break;
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            Result.Fill(.0f);
            switch (operation)
            {
                case MatOperation.Multiplication:  // vectors/matrices have to be always in the correct dimesions!
                    if (A.Count == 1) // valueA * B
                    {
                        A.SafeCopyToHost();
                        MyCublasFactory.Instance.Axpy(A.Host[0], B.GetDevice(callee), 1, Result.GetDevice(callee), 1);
                    }
                    else if (B.Count == 1) // A * valueB
                    {
                        B.SafeCopyToHost();
                        MyCublasFactory.Instance.Axpy(B.Host[0], A.GetDevice(callee), 1, Result.GetDevice(callee), 1);
                    }
                    else /// another executions...
                    {
                        Run(operation, A.GetDevice(callee), A.Count, A.ColumnHint, B.GetDevice(callee), B.Count, B.ColumnHint, Result.GetDevice(callee), Result.Count, Result.ColumnHint);
                    }
                    break;
                case MatOperation.DotProd:
                    Run(operation, A.GetDevice(callee), A.Count, A.ColumnHint, B.GetDevice(callee), B.Count, B.ColumnHint, Result.GetDevice(callee), Result.Count, Result.ColumnHint);
                    break;
                default:
                    MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cublas for undefined MatOperation");
                    break;
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result)
        {
            int itmp;
            Result.Fill(.0f);
            switch (operation)
            {
                case MatOperation.MinIndex:
                    itmp = MyCublasFactory.Instance.Min(A.GetDevice(callee), 1);
                    Result.Fill((float)(itmp - 1));
                    break;
                case MatOperation.MaxIndex:
                    itmp = MyCublasFactory.Instance.Max(A.GetDevice(callee), 1);
                    Result.Fill((float)(itmp - 1));
                    break;
                case MatOperation.Norm2:
                    MyCublasFactory.Instance.Norm2(A.GetDevice(callee), 1, Result.GetDevice(callee));
                    break;
                case MatOperation.Normalize:
                    float nrm = MyCublasFactory.Instance.Norm2(A.GetDevice(callee), 1);
                    MyCublasFactory.Instance.Axpy(1 / nrm, A.GetDevice(callee), 1, Result.GetDevice(callee), 1);
                    break;
                case MatOperation.Minus:
                    MyCublasFactory.Instance.Axpy(-1.0f, A.GetDevice(callee), 1, Result.GetDevice(callee), 1);
                    break;
                case MatOperation.Copy:
                    MyCublasFactory.Instance.Copy(A.GetDevice(callee), 1, Result.GetDevice(callee), 1);
                    break;
                default:
                    MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cublas for undefined MatOperation");
                    break;
            }
        }

        public override void Run(MatOperation operation, MyMemoryBlock<float> A)
        {
            switch (operation)
            {
                case MatOperation.Minus:
                    MyCublasFactory.Instance.Scale(-1.0f, A.GetDevice(callee), 1);
                    break;
                case MatOperation.Normalize:
                    float nrm = MyCublasFactory.Instance.Norm2(A.GetDevice(callee), 1);
                    MyCublasFactory.Instance.Scale(1 / nrm, A.GetDevice(callee), 1);
                    break;
                default:
                    MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cublas for undefined MatOperation");
                    break;
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, float value, MyMemoryBlock<float> Result)
        {
            Result.Fill(.0f);
            switch (operation)
            {
                case MatOperation.Multiplication:
                    MyCublasFactory.Instance.Axpy(value, A.GetDevice(callee), 1, Result.GetDevice(callee), 1);
                    break;
                default:
                    MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cublas for undefined MatOperation");
                    break;
            }
        }


        public static MatOperation AvailableOperations()
        {
            return MatOperation.Multiplication | MatOperation.MinIndex | MatOperation.MaxIndex | MatOperation.DotProd | MatOperation.Norm2 | MatOperation.Normalize | MatOperation.Minus | MatOperation.Copy;
        }






        /// <summary>
        ///   HONZA:   THIS IS FIXED, TRY RUN WITH CUDA CudaDeviceVariable AND IF IT WORKS, REPLACE THIS FUCNTION BY NEW ONE :D
        /// Computes y = Ax or y = A'x if <paramref name="transposeA"/> is true.
        ///     ???? ADD IT INOT THE CODE !!!!!  ????
        /// </summary>
        public static void MatrixVectorMult(CudaDeviceVariable<float> A, CudaDeviceVariable<float> x, CudaDeviceVariable<float> y,
            int xDim, int yDim,
            bool transposeA = false)
        {
            if (!transposeA)
            {
                // Compute y = Ax
                MyCublasFactory.Instance.Gemv(Operation.NonTranspose,
                    yDim, xDim, 1.0f,
                    A, yDim,
                    x, 1,
                    0f, y, 1);
            }
            else
            {
                // Compute y = A'x
                MyCublasFactory.Instance.Gemv(Operation.Transpose,
                    xDim, yDim, 1.0f,
                    A, xDim,
                    x, 1,
                    0f, y, 1);
            }
        }
    }
}
