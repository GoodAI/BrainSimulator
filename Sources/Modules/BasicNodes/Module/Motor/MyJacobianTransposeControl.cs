using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Matrix;
using System.ComponentModel;
using System.Drawing;
using YAXLib;

namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Computes set of torques that emulate effect of a virtual force applied to a point on body</summary>
    /// <description>Inverse kinematics method that uses transpose of Jacobian matrix to calculate joint torques that emulate effect of a virtual force applied to a point of a body in 3D coordinates<br />
    /// I/O:
    ///              <ul>
    ///                 <li>Anchors: Positions of joint anchors in a 3xN matrix, where each row contains X, Y, and Z coordinates for corresponding joint</li>
    ///                 <li>RotationAxes: 3xN matrix where each row is a 3D unit vector pointing along the direction of current axis of rotation for corresponding joint</li>
    ///                 <li>Point: X, Y, and Z coordintes of end effector where the force is to be applied</li>
    ///                 <li>Force: Virtual force 3D vector</li>
    ///                 <li>Output: Joint torques</li>
    ///              </ul>
    /// </description>
    [YAXSerializeAs("JacobianTransposeControl")]
    public class MyJacobianTransposeControl : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Anchors { get { return GetInput(0); } }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> RotationAxes { get { return GetInput(1); } }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Point { get { return GetInput(2); } }

        [MyInputBlock(3)]
        public MyMemoryBlock<float> Force { get { return GetInput(3); } }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<float> JacobianTranspose { get; private set; }

        public int COLUMNS;
        public int ROWS;

        /// <summary>Jacobian transpose method</summary>
        [Description("Transpose method"), MyTaskInfo(OneShot = false)]
        public class MyJacobianTransposeTask : MyTask<MyJacobianTransposeControl>
        {
            private MyVector3D m_point, m_anchor, m_axis, m_rateOfChange;
            private MyMatrixOps m_matrixOps;

            public override void Init(int nGPU)
            {
                m_point = new MyVector3D();
                m_anchor = new MyVector3D();
                m_axis = new MyVector3D();
                m_rateOfChange = new MyVector3D();
                m_matrixOps = new MyMatrixAutoOps(Owner, MatOperation.Multiplication);
            }

            public override void Execute()
            {
                Owner.Anchors.SafeCopyToHost();
                Owner.RotationAxes.SafeCopyToHost();
                Owner.Point.SafeCopyToHost();

                for (int i = 0; i < Owner.ROWS; i++)
                {
                    m_point.X = Owner.Point.Host[0];
                    m_point.Y = Owner.Point.Host[1];
                    m_point.Z = Owner.Point.Host[2];
                    
                    m_anchor.X = Owner.Anchors.Host[3 * i];
                    m_anchor.Y = Owner.Anchors.Host[3 * i + 1];
                    m_anchor.Z = Owner.Anchors.Host[3 * i + 2];

                    m_axis.X = Owner.RotationAxes.Host[3 * i];
                    m_axis.Y = Owner.RotationAxes.Host[3 * i + 1];
                    m_axis.Z = Owner.RotationAxes.Host[3 * i + 2];
                    
                    m_rateOfChange = MyVector3D.CrossProduct(m_axis, m_point - m_anchor);
                    
                    Owner.JacobianTranspose.Host[3 * i] = (float) m_rateOfChange.X;
                    Owner.JacobianTranspose.Host[3 * i + 1] = (float) m_rateOfChange.Y;
                    Owner.JacobianTranspose.Host[3 * i + 2] = (float) m_rateOfChange.Z;
                }

                for (int i = 0; i < Owner.JacobianTranspose.Count; i++)
                {
                    if (float.IsNaN(Owner.JacobianTranspose.Host[i]))
                    {
                        Owner.JacobianTranspose.Host[i] = 0;
                    }
                }

                Owner.JacobianTranspose.SafeCopyToDevice();

                m_matrixOps.Run(Matrix.MatOperation.Multiplication, Owner.JacobianTranspose, Owner.Force, Owner.Output);
            }
        }

        public MyJacobianTransposeTask jacobianTransposeTask { get; protected set; }

        public override void UpdateMemoryBlocks()
        {
            if (Anchors != null && RotationAxes != null && Point != null && Force != null)
            {
                COLUMNS = 3; //works only in 3 dimensions
                ROWS = Anchors.Count / COLUMNS;

                JacobianTranspose.Count = COLUMNS * ROWS;
                JacobianTranspose.ColumnHint = COLUMNS;

                Output.Count = ROWS;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Anchors != null && Anchors.ColumnHint == 3, this, "Anchors must be a 3xN matrix");
            validator.AssertError(RotationAxes != null && RotationAxes.ColumnHint == 3, this, "RotationAxes must be a 3xN matrix");
            validator.AssertError(Anchors != null && RotationAxes != null && Anchors.Count == RotationAxes.Count, this, "Number of anchors and rotation axes must be the same");
            validator.AssertError(Point != null && Point.Count == 3, this, "Point must be a 3 dimensional vector");
            validator.AssertError(Force != null && Force.Count == 3, this, "Force must be a 3 dimensional vector");
        }
    }

    internal class MyVector3D
    {
        public float X, Y, Z;

        public MyVector3D() { }
        public MyVector3D(MyVector3D vector)
        {
            X = vector.X;
            Y = vector.Y;
            Z = vector.Z;
        }
        public MyVector3D(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static MyVector3D operator -(MyVector3D a, MyVector3D b)
        {
            return new MyVector3D(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        public static MyVector3D CrossProduct(MyVector3D a, MyVector3D b)
        {
            return new MyVector3D((a.Y * b.Z) - (b.Y * a.Z), (a.Z * b.X) - (b.Z * a.X), (a.X * b.Y) - (b.X * a.Y));
        }
    }
}
