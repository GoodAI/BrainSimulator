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

namespace BrainSimulator.Observers
{
    /// <summary>
    /// Observers SRPs (Stochastic Return Predictors) in the HARM (each one has name, own motivation, promoted variable etc..).
    /// </summary>
    public class MySRPObserver : MyAbstractQLearningObserver<MyDiscreteHarmNode>
    {
        [MyBrowsable, Category("Mode"),
        Description("Show names of variables that are controlled by this strategy, together with the current motivation"),
        YAXSerializableField(DefaultValue = true)]
        public bool ShowSRPNames { get; set; }

        [YAXSerializableField(DefaultValue = 2)]
        private int m_srpVariableIndex;

        [MyBrowsable, Category("Q Selection"),
        Description("Index of the abstract action to be shown (in order in which the actions were discovered)")]
        public int AbstractActionIndex
        {
            get { return m_srpVariableIndex; }
            set
            {
                m_srpVariableIndex = value;
                TriggerReset();
            }
        }

        public MyRootDecisionSpace Rds { get; set; }

        protected override void Execute()
        {
            if (m_qMatrix != null)
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

            MyStochasticReturnPredictor srp = null;

            if (AbstractActionIndex < Target.Rds.VarManager.MAX_VARIABLES)
            {
                srp = (MyStochasticReturnPredictor)Target.Vis.GetPredictorNo(AbstractActionIndex);
            }

            if (srp != null)
            {
                Target.Vis.ReadTwoDimensions(ref m_qMatrix, ref m_qMatrixActions, srp, XAxisVariableIndex, YAxisVariableIndex, ShowCurrentMotivations);
            }

            if (lastQMatrix != m_qMatrix)
            {
                TriggerReset();
            }
            else if (m_qMatrix != null && MatrixSizeOK())
            {
                m_plotValues.CopyToDevice(m_qMatrix);
                m_actionIndices.CopyToDevice(m_qMatrixActions);

                if (ShowSRPNames && srp != null)
                {
                    //Set texture size, it will trigger texture buffer reallocation
                    TextureWidth = LABEL_PIXEL_WIDTH * m_qMatrix.GetLength(0);
                    TextureHeight = LABEL_PIXEL_WIDTH * m_qMatrix.GetLength(1);

                    String label = srp.GetLabel() + " M:" + srp.GetMyTotalMotivation();

                    MyDrawStringHelper.DrawString(label, 0, 0, 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight);
                }
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

            MyStochasticReturnPredictor srp = null;

            if (AbstractActionIndex < Target.Rds.VarManager.MAX_VARIABLES)
            {
                srp = (MyStochasticReturnPredictor)Target.Vis.GetPredictorNo(AbstractActionIndex);
            }

            if (srp == null)
            {
                m_qMatrix = null;
                TextureWidth = 0;
                TextureHeight = 0;
            }
            else
            {
                Target.Vis.ReadTwoDimensions(ref m_qMatrix, ref m_qMatrixActions, srp, XAxisVariableIndex, YAxisVariableIndex, ShowCurrentMotivations);
                if (MatrixSizeOK())
                {
                    DrawDataToGpu();
                }
            }
        }
    }
}

