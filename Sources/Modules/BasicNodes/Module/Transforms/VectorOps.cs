using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Matrix;
using System;
using System.Diagnostics;

namespace GoodAI.BasicNodes.Transforms
{
    public class VectorOps
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
        private readonly MyMatrixAutoOps m_matOperation;
        private readonly MyMemoryBlock<float> m_temp;

        public static float DegreeToRadian(float angle)
        {
            return (float)Math.PI * angle / 180;
        }

        public static float RadianToDegree(float angle)
        {
            return angle * 180 / (float)Math.PI;
        }

        public VectorOps(MyWorkingNode caller, VectorOperation operations, MyMemoryBlock<float> tempBlock)
        {
            m_caller = caller;
            m_operations = operations;
            m_temp = tempBlock;

            MatOperation mat_ops = MatOperation.None;

            if (m_operations.HasFlag(VectorOperation.Rotate))
            {
                Debug.Assert(tempBlock.Count >= 4, "Temporary memory block has to be large at least 4 items when using Rotate operation");
                mat_ops |= MatOperation.Multiplication;
            }

            if (m_operations.HasFlag(VectorOperation.Angle))
                mat_ops |= MatOperation.DotProd;

            if (m_operations.HasFlag(VectorOperation.DirectedAngle))
            {
                mat_ops |= MatOperation.Multiplication | MatOperation.DotProd;
                m_operations |= VectorOperation.Angle | VectorOperation.Rotate;
            }
                

            m_matOperation = new MyMatrixAutoOps(caller, mat_ops);
        }

        public void Run(VectorOperation operation,
            MyMemoryBlock<float> a,
            MyMemoryBlock<float> b,
            MyMemoryBlock<float> result)
        {
            if (!Validate(operation, a.Count, b.Count))
                return;

            switch (operation)
            {
                case VectorOperation.Rotate:
                {
                    b.SafeCopyToHost();
                    float rads = DegreeToRadian(b.Host[0]);
                    float[] transform = { (float)Math.Cos(rads), -(float)Math.Sin(rads), (float)Math.Sin(rads), (float)Math.Cos(rads) };
                    Array.Copy(transform, m_temp.Host, transform.Length);
                    m_temp.SafeCopyToDevice();
                    m_matOperation.Run(MatOperation.Multiplication, m_temp, a, result);
                }
                break;

                case VectorOperation.Angle:
                {
                    m_matOperation.Run(MatOperation.DotProd, a, b, result);
                    result.SafeCopyToHost();
                    float dotProd = result.Host[0];
                    float angle = RadianToDegree((float)Math.Acos(dotProd));
                    result.Fill(0);
                    result.Host[0] = angle;
                    result.SafeCopyToDevice();
                }
                break;

                case VectorOperation.DirectedAngle:
                {
                    result.Host[0] = -90;
                    result.SafeCopyToDevice();
                    Run(VectorOperation.Rotate, a, result, result);
                    result.CopyToMemoryBlock(m_temp, 0, 0, result.Count);

                    m_matOperation.Run(MatOperation.DotProd, a, b, result);
                    result.SafeCopyToHost();
                    float dotProd = result.Host[0];
                    float angle;
                    if (Math.Abs(Math.Abs(dotProd) - 1) < 1E-4)
                        angle = 0;
                    else
                        angle = RadianToDegree((float)Math.Acos(dotProd));

                    m_matOperation.Run(MatOperation.DotProd, m_temp, b, result);
                    result.SafeCopyToHost();
                    float perpDotProd = result.Host[0];

                    if (perpDotProd > 0)
                        angle *= -1;
                    result.Fill(0);
                    result.Host[0] = angle;
                    result.SafeCopyToDevice();
                }
                break;
            }
        }

        private bool Validate(VectorOperation operation, int sizeA, int sizeB)
        {
            if (operation == VectorOperation.None)
                return false;

            if ((operation & m_operations) == 0)
            {
                MyLog.WARNING.WriteLine("Trying to execute an uninitialized vector operation. Owner: " + m_caller.Name);
                return false;
            }

            if (operation != VectorOperation.Rotate && sizeA != sizeB)
            {
                MyLog.ERROR.WriteLine("Vectors have to be the same length for this operation. Owner: " + m_caller.Name);
                return false;
            }

            return true;
        }
    }
}
