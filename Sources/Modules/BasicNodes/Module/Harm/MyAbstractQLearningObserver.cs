using GoodAI.Core.Observers.Helper;
using GoodAI.Modules.Retina;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Modules.Harm;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core;

namespace GoodAI.Modules.Observers
{
    /// <author>GoodAI</author>
    /// <meta>df,jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Observes valdata stored in the QMatrix.
    /// </summary>
    /// <typeparam name="T">Node which uses DiscreteQLearnin to be observed</typeparam>
    public abstract class MyAbstractQLearningObserver<T> : MyNodeObserver<T> where T : MyAbstractDiscreteQLearningNode
    {
        [MyBrowsable, Category("Mode"),
        Description("Set a reasonable value for good color scaling.")]
        [YAXSerializableField(DefaultValue = 0.003f)]
        public float MaxQValue { get; set; }

        [MyBrowsable, Category("Mode"),
        Description("Observe utility colors which are already scaled by the current motivation.")]
        [YAXSerializableField(DefaultValue = true)]
        public bool ShowCurrentMotivations { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        private int m_xAxisVariableIndex;

        [MyBrowsable, Category("Q Selection"),
        Description("Index of variable to be displayed on the X axis")]
        [YAXSerializableField(DefaultValue = 0)]
        public int XAxisVariableIndex
        {
            get { return m_xAxisVariableIndex; }
            set
            {
                m_xAxisVariableIndex = value;
                TriggerReset();
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_yVariableIndex;

        [MyBrowsable, Category("Q Selection"),
        Description("Index of variable to be displayed on the Y axis")]
        [YAXSerializableField(DefaultValue = 1)]
        public int YAxisVariableIndex
        {
            get { return m_yVariableIndex; }
            set
            {
                m_yVariableIndex = value;
                TriggerReset();
            }
        }

        // if the user attempts to draw a memory block that is too big, do not swhow it (out of memory error)
        public static readonly int QMATRIX_MAX_SIZE = 50000;
        private int prevSx = -1, prevSy = -1;

        protected MyCudaKernel m_vertexKernel;
        protected MyCudaKernel m_setKernel;

        protected float[,] m_qMatrix = null;
        protected int[,] m_qMatrixActions = null;
        protected int numOfActions = 0;

        public const int LABEL_PIXEL_WIDTH = 32;

        protected CudaDeviceVariable<float> m_plotValues;
        protected CudaDeviceVariable<int> m_actionIndices;
        protected CudaDeviceVariable<uint> m_actionLabels;

        public MyAbstractQLearningObserver()
        {
            m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Harm\MatrixQLearningKernel", "createTexture");
            m_vertexKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Harm\MatrixQLearningKernel", "crate3Dplot");
            m_setKernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Common\SetKernel");

            TriggerReset();
        }

        /// <summary>
        /// Is not OK either if both dimensions have zero size or the total size of matrix is too big for GPU mem
        /// </summary>
        /// <returns></returns>
        protected bool MatrixSizeOK()
        {
            int size = m_qMatrix.GetLength(0) * m_qMatrix.GetLength(1);
            bool bothZero = m_qMatrix.GetLength(0) == 0 && m_qMatrix.GetLength(1) == 0; 
 
            if (size >= QMATRIX_MAX_SIZE)
            {
                if (SizeChanged())
                {
                    MyLog.DEBUG.WriteLine("Observer: you are trying to display too big data!"
                        + " the matrix size is: " + m_qMatrix.GetLength(0) + "x" + m_qMatrix.GetLength(1)
                        + ", max allowed size is: " + QMATRIX_MAX_SIZE);
                }
            }
            return size < QMATRIX_MAX_SIZE && !bothZero;
        }

        private bool SizeChanged()
        {
            bool changed = prevSx != m_qMatrix.GetLength(0) || prevSy != m_qMatrix.GetLength(1);

            prevSx = m_qMatrix.GetLength(0);
            prevSy = m_qMatrix.GetLength(1);
            return changed;
        }

        protected void DrawDataToGpu()
        {
            if (m_qMatrix != null && m_qMatrix.Length > 0)
            {
                //Set texture size, it will trigger texture buffer reallocation
                TextureWidth = LABEL_PIXEL_WIDTH * m_qMatrix.GetLength(0);
                TextureHeight = LABEL_PIXEL_WIDTH * m_qMatrix.GetLength(1);

                if (m_plotValues != null)
                {
                    m_plotValues.Dispose();
                }

                if (m_actionIndices != null)
                {
                    m_actionIndices.Dispose();
                }

                m_plotValues = new CudaDeviceVariable<float>(m_qMatrix.Length);
                m_plotValues.CopyToDevice(m_qMatrix);

                m_actionIndices = new CudaDeviceVariable<int>(m_qMatrixActions.Length);
                m_actionIndices.CopyToDevice(m_qMatrixActions);

                if (ViewMode == ViewMethod.Orbit_3D)
                {
                    //Set vertex data size, it will trigger vertex buffer reallocation
                    VertexDataSize = m_qMatrix.Length * 20 * 3;
                    VertexDataSize += m_qMatrix.Length * 4 * 2;

                    Vector3 translation = new Vector3(-m_qMatrix.GetLength(0) * 0.05f, 0, -m_qMatrix.GetLength(1) * 0.05f);

                    //top side, textured
                    Shapes.Add(new MyBufferedPrimitive(PrimitiveType.Quads, m_qMatrix.Length * 4, MyVertexAttrib.Position | MyVertexAttrib.TexCoord)
                    {
                        VertexOffset = 0,
                        TexCoordOffset = m_qMatrix.Length * 20 * 3, //tex coord are located behind all vertices
                        Translation = translation
                    });

                    //left side, sides are separated due different color. Colors are not stored in the buffer
                    Shapes.Add(new MyBufferedPrimitive(PrimitiveType.Quads, m_qMatrix.Length * 4, MyVertexAttrib.Position)
                    {
                        VertexOffset = m_qMatrix.Length * 4 * 3,
                        BaseColor = Color.FromArgb(160, 180, 160),
                        Translation = translation
                    });

                    //far side
                    Shapes.Add(new MyBufferedPrimitive(PrimitiveType.Quads, m_qMatrix.Length * 4, MyVertexAttrib.Position)
                    {
                        VertexOffset = m_qMatrix.Length * 8 * 3,
                        BaseColor = Color.FromArgb(140, 160, 140),
                        Translation = translation
                    });

                    //near side
                    Shapes.Add(new MyBufferedPrimitive(PrimitiveType.Quads, m_qMatrix.Length * 4, MyVertexAttrib.Position)
                    {
                        VertexOffset = m_qMatrix.Length * 12 * 3,
                        BaseColor = Color.FromArgb(180, 200, 180),
                        Translation = translation
                    });

                    //right side
                    Shapes.Add(new MyBufferedPrimitive(PrimitiveType.Quads, m_qMatrix.Length * 4, MyVertexAttrib.Position)
                    {
                        VertexOffset = m_qMatrix.Length * 16 * 3,
                        BaseColor = Color.FromArgb(160, 180, 160),
                        Translation = translation
                    });
                }
                else
                {
                    Shapes.Add(new MyDefaultShape());
                }
            }
        }
    }
}
