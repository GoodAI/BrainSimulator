using GoodAI.BasicNodes.DiscreteRL.Observers;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Utils;
using GoodAI.Modules.DiscreteRL.Observers;
using GoodAI.Modules.Harm;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.DiscreteRL.Observers
{

    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Observers SRPs (Stochastic Return Predictors) in the HARM (each one has name, own motivation, promoted variable etc..).
    /// </summary>
    public class DiscreteSRPObserver : DiscretePolicyObserver<MyDiscreteHarmNode>
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

        private CudaDeviceVariable<float> m_StringDeviceBuffer;

        protected override void Execute()
        {
            if (m_qMatrix != null)
            {
                m_kernel.SetupExecution(TextureWidth * TextureHeight);
                m_kernel.Run(m_plotValues.DevicePointer, m_actionIndices.DevicePointer, m_actionLabels.DevicePointer, numOfActions, LABEL_PIXEL_WIDTH, LABEL_PIXEL_WIDTH, 0f, MaxUtilityValue, m_qMatrix.GetLength(0), m_qMatrix.GetLength(1), VBODevicePointer);

                if (ViewMode == ViewMethod.Orbit_3D)
                {
                    m_vertexKernel.SetupExecution(m_qMatrix.Length);
                    m_vertexKernel.Run(m_plotValues.DevicePointer, 0.1f, m_qMatrix.GetLength(0), m_qMatrix.GetLength(1), MaxUtilityValue, VertexVBODevicePointer);
                }
            }

            float[,] lastQMatrix = m_qMatrix;

            MyStochasticReturnPredictor srp = null;
            
            srp = (MyStochasticReturnPredictor)Target.Vis.GetPredictorNo(AbstractActionIndex);
            
            if (srp != null)
            {
                Target.ReadTwoDimensions(ref m_qMatrix, ref m_qMatrixActions, XAxisVariableIndex, YAxisVariableIndex, ApplyInnerScaling, AbstractActionIndex);
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

                    MyDrawStringHelper.String2Index(label, m_StringDeviceBuffer);
                    MyDrawStringHelper.DrawStringFromGPUMem(m_StringDeviceBuffer, 0, 0, 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight, 0, label.Length);
                    //MyDrawStringHelper.DrawString(label, 0, 0, 0, 0x999999, VBODevicePointer, TextureWidth, TextureHeight);
                }
            }
        }

        protected override void Reset()
        {
            m_StringDeviceBuffer = new CudaDeviceVariable<float>(1000);
            m_StringDeviceBuffer.Memset(0);

            List<String> actions = Target.GetActionLabels();

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
                    MyDrawStringHelper.String2Index(actions[i], m_StringDeviceBuffer);
                    MyDrawStringHelper.DrawStringFromGPUMem(m_StringDeviceBuffer, i * LABEL_PIXEL_WIDTH + 5, 8, 0, 0xFFFFFFFF, m_actionLabels.DevicePointer, LABEL_PIXEL_WIDTH * actions.Count, LABEL_PIXEL_WIDTH, 0, actions[i].Length);
                }

                numOfActions = actions.Count;
            }

            MyStochasticReturnPredictor srp = null;
            
            srp = (MyStochasticReturnPredictor)Target.Vis.GetPredictorNo(AbstractActionIndex);
            
            if (srp == null)
            {
                m_qMatrix = null;
                TextureWidth = 0;
                TextureHeight = 0;
            }
            else
            {
                Target.ReadTwoDimensions(ref m_qMatrix, ref m_qMatrixActions, XAxisVariableIndex, YAxisVariableIndex, ApplyInnerScaling, AbstractActionIndex);
                
                if (MatrixSizeOK())
                {
                    DrawDataToGpu();
                }
            }
        }
    }
}

