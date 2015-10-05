using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.Matrix;
using System;

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

        private readonly MyWorkingNode caller;
        private readonly VectorOperation operations;
        private readonly MyMatrixAutoOps mat_operation;
        private readonly MyMemoryBlock<float> temp;

        public VectorOps(MyWorkingNode caller, VectorOperation operations, MyMemoryBlock<float> tempBlock)
        {
            this.caller = caller;
            this.operations = operations;
            temp = tempBlock;

            MatOperation mat_ops = MatOperation.None;

            if (this.operations.HasFlag(VectorOperation.Rotate))
                mat_ops |= MatOperation.Multiplication;

            if (this.operations.HasFlag(VectorOperation.Angle))
                mat_ops |= MatOperation.DotProd;

            if (this.operations.HasFlag(VectorOperation.DirectedAngle))
            {
                mat_ops |= MatOperation.Multiplication | MatOperation.DotProd;
                this.operations |= VectorOperation.Angle | VectorOperation.Rotate;
            }
                

            mat_operation = new MyMatrixAutoOps(caller, mat_ops);
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
                    float rads = b.Host[0] * (float)Math.PI / 180;
                    float[] transform = { (float)Math.Cos(rads), -(float)Math.Sin(rads), (float)Math.Sin(rads), (float)Math.Cos(rads) };
                    Array.Copy(transform, temp.Host, transform.Length);
                    temp.SafeCopyToDevice();
                    mat_operation.Run(MatOperation.Multiplication, temp, a, result);
                }
                break;

                case VectorOperation.Angle:
                {
                    mat_operation.Run(MatOperation.DotProd, a, b, result);
                    result.SafeCopyToHost();
                    float dotProd = result.Host[0];
                    float angle = (float)Math.Acos(dotProd) * 180 / (float)Math.PI;
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
                    result.CopyToMemoryBlock(temp, 0, 0, result.Count);

                    mat_operation.Run(MatOperation.DotProd, a, b, result);
                    result.SafeCopyToHost();
                    float dotProd = result.Host[0];
                    float angle;
                    if (dotProd == 1)
                        angle = 0;
                    else
                        angle = (float)Math.Acos(dotProd) * 180 / (float)Math.PI;

                    mat_operation.Run(MatOperation.DotProd, temp, b, result);
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

            if ((operation & operations) == 0)
            {
                MyLog.WARNING.WriteLine("Trying to execute an uninitialized vector operation. Owner: " + caller.Name);
                return false;
            }

            if (operation != VectorOperation.Rotate && sizeA != sizeB)
            {
                MyLog.ERROR.WriteLine("Vectors have to be the same length for this operation. Owner: " + caller.Name);
                return false;
            }

            return true;
        }
    }
}
