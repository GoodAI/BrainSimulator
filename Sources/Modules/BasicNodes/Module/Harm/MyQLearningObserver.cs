using BrainSimulator.Observers.Helper;
using BrainSimulator.Retina;
using BrainSimulator.Task;
using BrainSimulator.Utils;
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
using BrainSimulator.Harm;
using BrainSimulator.Observers;

namespace BrainSimulator.Observers
{
    /// <summary>
    /// Observes only flat QLearning node.
    /// </summary>
    public class MyQLearningObserver : MyAbstractQLearningObserver<MyDiscreteQLearningNode>
    {
        protected override void Execute()
        {

            if (m_qMatrix != null && MatrixSizeOK())
            {
                m_kernel.SetupExecution(TextureWidth * TextureHeight);
                m_kernel.Run(m_plotValues.DevicePointer, m_actionIndices.DevicePointer, m_actionLabels.DevicePointer, numOfActions, LABEL_PIXEL_WIDTH, LABEL_PIXEL_WIDTH, 0f, MaxQValue, m_qMatrix.GetLength(0), m_qMatrix.GetLength(1), VBODevicePointer);

                if (ViewMode == ViewMethod.Orbit_3D)
                {
                    m_vertexKernel.SetupExecution(m_qMatrix.Length);
                    m_vertexKernel.Run(m_plotValues.DevicePointer, 0.1f, m_qMatrix.GetLength(0), m_qMatrix.GetLength(1), MaxQValue, VertexVBODevicePointer);
                }
            }

            float[,] lastQMatrix = m_qMatrix;
            Target.Vis.ReadTwoDimensions(ref m_qMatrix, ref m_qMatrixActions, XAxisVariableIndex, YAxisVariableIndex, ShowCurrentMotivations);

            if (lastQMatrix != m_qMatrix)
            {
                TriggerReset();
            }
            else if (m_qMatrix != null && base.MatrixSizeOK())
            {
                m_plotValues.CopyToDevice(m_qMatrix);
                m_actionIndices.CopyToDevice(m_qMatrixActions);
            }
        }

        protected override void Reset()
        {
            List<MyMotivatedAction> actions = Target.Rds.ActionManager.Actions;

            if (numOfActions < actions.Count)
            {
                if (m_actionLabels != null)
                {
                    m_actionLabels.Dispose();
                }

                m_actionLabels = new CudaDeviceVariable<uint>(actions.Count * LABEL_PIXEL_WIDTH * LABEL_PIXEL_WIDTH);
                m_actionLabels.Memset(0);

                for (int i = 0; i < actions.Count; i++)
                {
                    MyDrawStringHelper.DrawString(actions[i].GetLabel(), i * LABEL_PIXEL_WIDTH + 5, 8, 0, 0xFFFFFFFF, m_actionLabels.DevicePointer, LABEL_PIXEL_WIDTH * actions.Count, LABEL_PIXEL_WIDTH);
                }

                numOfActions = actions.Count;
            }
            Target.Vis.ReadTwoDimensions(ref m_qMatrix, ref m_qMatrixActions, XAxisVariableIndex, YAxisVariableIndex, ShowCurrentMotivations);

            if (MatrixSizeOK())
            {
                DrawDataToGpu();
            }
        }
    }
}
