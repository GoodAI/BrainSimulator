using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Matrix;
using System;

namespace GoodAI.BasicNodes.Transforms
{
    public class MyVectorOps
    {
        [Flags]
        public enum VectorOperation
        {
            None = 0,

            Rotate = 1 << 1,
            Angle = 1 << 2,
            DirectedAngle = 1 << 3,
        }

        private readonly MyWorkingNode m_caller;
        private readonly VectorOperation m_operations;
        private readonly MyMatrixAutoOps mat_operation;
        private readonly MyMemoryBlock<float> m_temp;

        public MyVectorOps(MyWorkingNode caller, VectorOperation operations, MyMemoryBlock<float> tempBlock)
        {
            m_caller = caller;
            m_operations = operations;
            m_temp = tempBlock;

            MatOperation mat_ops = MatOperation.None;

            if (operations.HasFlag(VectorOperation.Rotate))
                mat_ops |= MatOperation.Multiplication;

            if (operations.HasFlag(VectorOperation.Angle))
                mat_ops |= MatOperation.DotProd;

            if (operations.HasFlag(VectorOperation.DirectedAngle))
                mat_ops |= MatOperation.Multiplication | MatOperation.DotProd;

            mat_operation = new MyMatrixAutoOps(m_caller, mat_ops);
        }

        public void Run(VectorOperation operation,
            MyMemoryBlock<float> A,
            MyMemoryBlock<float> B,
            MyMemoryBlock<float> Result)
        {
            if (!Validate(operation))
                return;

            switch (operation)
            {
                case VectorOperation.Rotate:
                {
                    B.SafeCopyToHost();
                    float rads = B.Host[0] * (float)Math.PI / 180;
                    float[] transform = new float[] { (float)Math.Cos(rads), -(float)Math.Sin(rads), (float)Math.Sin(rads), (float)Math.Cos(rads) };
                    Array.Copy(transform, m_temp.Host, transform.Length);
                    m_temp.SafeCopyToDevice();
                    mat_operation.Run(MatOperation.Multiplication, m_temp, A, Result);
                }
                break;

                case VectorOperation.Angle:
                {
                    mat_operation.Run(MatOperation.DotProd, A, B, Result);
                    Result.SafeCopyToHost();
                    float dotProd = Result.Host[0];
                    float angle = (float)Math.Acos(dotProd) * 180 / (float)Math.PI;
                    Result.Fill(0);
                    Result.Host[0] = angle;
                    Result.SafeCopyToDevice();
                }
                break;

                case VectorOperation.DirectedAngle:
                {
                    float rads = -(float)Math.PI / 2;
                    float[] transform = new float[] { (float)Math.Cos(rads), -(float)Math.Sin(rads), (float)Math.Sin(rads), (float)Math.Cos(rads) };
                    Array.Copy(transform, m_temp.Host, transform.Length);
                    m_temp.SafeCopyToDevice();
                    mat_operation.Run(MatOperation.Multiplication, m_temp, A, Result);
                    Result.CopyToMemoryBlock(m_temp, 0, 0, Result.Count);

                    mat_operation.Run(MatOperation.DotProd, A, B, Result);
                    Result.SafeCopyToHost();
                    float dotProd = Result.Host[0];
                    float angle;
                    if (dotProd == 1)
                        angle = 0;
                    else
                        angle = (float)Math.Acos(dotProd) * 180 / (float)Math.PI;

                    mat_operation.Run(MatOperation.DotProd, m_temp, B, Result);
                    Result.SafeCopyToHost();
                    float perpDotProd = Result.Host[0];

                    if (perpDotProd > 0)
                        angle *= -1;
                    Result.Fill(0);
                    Result.Host[0] = angle;
                    Result.SafeCopyToDevice();
                }
                break;
            }
        }

        private bool Validate(VectorOperation operation)
        {
            if (operation == VectorOperation.None)
                return false;

            if ((operation & m_operations) == 0)
            {
                MyLog.WARNING.WriteLine("Trying to execute an uninitialized vector operation. Owner: " + m_caller.Name);
                return false;
            }

            return true;
        }
    }
}
