using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.ToyWorld;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Single pixel RL 2")]
    public class LTSinglePixelRL2 : LTSinglePixelRL
    {
        protected bool m_targetWithAction { get; set; }
        protected bool m_previousTargetWithAction { get; set; }

        public LTSinglePixelRL2() : this(null) { }

        public LTSinglePixelRL2(SchoolWorld w)
            : base(w)
        {
        }

        public override void PresentNewTrainingUnit()
        {
            m_targetsShown = 0;
            m_targetsDetected = 0;
            m_targetsMisdetected = 0;
            m_previousObjectType = ObjectType.None;
            m_currentObjectType = ObjectType.None;
            m_targetWithAction = true;
            m_previousTargetWithAction = true;
            m_object = null;

            // first step is different: always empty
            CreateObject(ObjectType.Empty);
        }

        public void NextTargetAction(bool doAction)
        {
            m_previousTargetWithAction = m_targetWithAction;
            m_targetWithAction = doAction;
        }

        public override void ExecuteStepBeforeEvaluation()
        {
            if (m_object != null)
                WrappedWorld.RemoveGameObject(m_object);
                
            if(m_currentObjectType == ObjectType.BeforeTarget)
            {
                NextTargetAction(m_targetWithAction);
                CreateObject(ObjectType.Target);
                m_targetsShown++;
            }
            else if (LearningTaskHelpers.FlipBiasedCoin(m_rndGen, 0.33f))
            {
                if (LearningTaskHelpers.FlipBiasedCoin(m_rndGen, 0.5f))
                {
                    NextTargetAction(false);
                    CreateObject(ObjectType.BeforeTarget);
                }
                else
                {
                    NextTargetAction(m_targetWithAction);
                    CreateObject(ObjectType.Target);
                    m_targetsShown++;
                }
            }
            else
            {
                NextTargetAction(true);
                CreateObject(ObjectType.Empty);
            }
        }

        protected override void SinglePixelRLEvaluateStep()
        {
            SchoolWorld.ActionInput.SafeCopyToHost();
            bool wasTargetDetected = SchoolWorld.ActionInput.Host[ControlMapper.Idx("forward")] != 0;
            bool wasTargetPresent = (m_previousObjectType == ObjectType.Target) && m_previousTargetWithAction;

            if (wasTargetDetected && wasTargetPresent)
            {
                m_targetsDetected++;
                WrappedWorld.Reward.Host[0] = 1f;
            }
            else if (m_previousObjectType != ObjectType.None)
            {
                if (wasTargetDetected && !wasTargetPresent)
                {
                    m_targetsMisdetected++;
                    WrappedWorld.Reward.Host[0] = -1f;
                }
                else if (wasTargetPresent && !wasTargetDetected)
                {
                    WrappedWorld.Reward.Host[0] = -1f;
                }
                else
                {
                    WrappedWorld.Reward.Host[0] = 0f;
                    if(m_previousObjectType == ObjectType.Target && !m_previousTargetWithAction && !wasTargetDetected)
                    {
                        m_targetsDetected++;
                    }
                }
            }
            else
            {
                WrappedWorld.Reward.Host[0] = 0;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            int correction = m_currentObjectType == ObjectType.Target ? 1 : 0;

            wasUnitSuccessful = (m_targetsShown - correction == m_targetsDetected && m_targetsMisdetected == 0);

            bool acted = SchoolWorld.ActionInput.Host[ControlMapper.Idx("forward")] != 0;
            MyLog.Writer.WriteLine(MyLogLevel.WARNING, "did act: " + acted + ", wasTarget: " + (m_previousObjectType == ObjectType.Target) +
                ", was allowed to act: " + m_previousTargetWithAction);

            MyLog.Writer.WriteLine(MyLogLevel.WARNING, "m_targetsShown:" + m_targetsShown + ", m_targetsDetected: " + m_targetsDetected +
                "correction: " + correction + ", targetsMisdetected: " + m_targetsMisdetected);

            return m_targetsShown - correction == m_targetsPerTU;
            // - correction, because when ending, ExecuteStep is done before the last EvaluateStep - the ExecuteStep may prepare a new state which should be ignored by EvaluateStep
        }

    }
}
