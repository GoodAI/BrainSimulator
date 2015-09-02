using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;




namespace GoodAI.Modules.Matrix
{




    /// <summary>
    /// 
    /// </summary>
    public class MyMatrixCPUOps : MyMatrixOps
    {
        public MyMatrixCPUOps(MyWorkingNode callee=null, MatOperation operation = 0, MyMemoryBlock<float> A = null, MyMemoryBlock<float> tmp = null)
        {
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> B, MyMemoryBlock<float> Result)
        {
            Result.Fill(.0f);
            switch (operation)
            {
                case MatOperation.EuclidDist:
                    if (B.Count == A.ColumnHint)
                    {
                        A.SafeCopyToHost();
                        B.SafeCopyToHost();
                        for (int row = 0; row < A.Count / A.ColumnHint; row++)
                        {
                            Result.Host[row] = 0;
                            for (int Bindex = 0; Bindex < B.Count; Bindex++)
                            {
                                Result.Host[row] += (B.Host[Bindex] - A.Host[A.ColumnHint * row + Bindex]) * (B.Host[Bindex] - A.Host[A.ColumnHint * row + Bindex]);
                            }
                            Result.Host[row] = (float)Math.Sqrt( (double) Result.Host[row] );
                            //System.Console.Write(" " + Result.Host[row]);
                        }
                        Result.SafeCopyToDevice();
                    }
                    break;
                default:
                    MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cpu mat ops. for undefined MatOperation");
                    break;
            }
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, MyMemoryBlock<float> Result)
        {
            MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cpu mat ops. for undefined MatOperation");
        }

        public override void Run(MatOperation operation, MyMemoryBlock<float> A)
        {
            MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cpu mat ops. for undefined MatOperation");
        }


        public override void Run(MatOperation operation, MyMemoryBlock<float> A, float value, MyMemoryBlock<float> Result)
        {
            MyLog.Writer.WriteLine(MyLogLevel.ERROR, "Trying to run cpu mat ops. for undefined MatOperation");
        }




        public static MatOperation AvailableOperations()
        {
            return MatOperation.EuclidDist;
        }


    }


}
