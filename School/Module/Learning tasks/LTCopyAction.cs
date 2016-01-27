using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTCopyAction : AbstractLearningTask<ManInWorld>
    {
        public const string STOP_REQUEST = "Stop request";

        // True if teacher agent can spawn on different places
        // Teacher should not cover agent
        public const string TEACHER_ON_DIFF_START_POSITION = "Teacher on diff start position";

        protected Random m_rndGen = new Random();
        protected int m_stepsSincePresented = 0;
        protected MovableGameObject m_agent;
        protected AgentsHistory m_agentsHistory;
        protected AbstractTeacherInWorld m_teacher;
        protected AgentsHistory m_teachersHistory;

        public LTCopyAction() { }

        public LTCopyAction(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { STOP_REQUEST, 0},
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.NOISE, 0 },
                { TEACHER_ON_DIFF_START_POSITION, 0},
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TEACHER_ON_DIFF_START_POSITION, 1);
            TSProgression.Add(TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 1.5f);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);
        }

        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();
            CreateTeacher();

            m_stepsSincePresented = 0;
            m_agentsHistory = new AgentsHistory();
            m_agentsHistory.Add(m_agent.X, m_agent.Y);
            m_teachersHistory = new AgentsHistory();
            m_teachersHistory.Add(m_teacher.X, m_teacher.Y);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (World.IsEmulatingUnitCompletion())
            {
                return World.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
            }
            else
            {
                m_stepsSincePresented++;

                if (DidTrainingUnitFail())
                {
                    wasUnitSuccessful = false;
                    return true;
                }

                if (!m_teacher.IsDone() && m_agent.isMoving())
                {
                    wasUnitSuccessful = false;
                    return true;
                }

                // save history for agent and teacher
                m_agentsHistory.Add(m_agent.X, m_agent.Y);
                m_teachersHistory.Add(m_teacher.X, m_teacher.Y);

                int numberOfTeachersSteps = m_teachersHistory.numberOfSteps();
                int numberOfAgentsSteps = m_agentsHistory.numberOfSteps();

                wasUnitSuccessful = false;
                // simple version of the task
                if (TSHints[LTCopyAction.STOP_REQUEST] == .0f)
                {
                    if (numberOfTeachersSteps == numberOfAgentsSteps && m_teacher.IsDone())
                    {
                        // compare step
                        wasUnitSuccessful = m_teachersHistory.CompareTo(m_agentsHistory, m_stepsSincePresented);
                        return true;
                    }
                }
                // hard version
                else
                {
                    if (m_stepsSincePresented >= 5)
                    {
                        // compare steps
                        wasUnitSuccessful = m_teachersHistory.CompareTo(m_agentsHistory, m_stepsSincePresented);
                        return true;
                    }
                }
                return false;
            }
        }

        protected bool DidTrainingUnitFail()
        {
            return m_stepsSincePresented > 2 * m_teacher.ActionsCount() + 3;
        }

        protected void CreateAgent()
        {
            m_agent = (World as RoguelikeWorld).CreateAgent();
        }

        protected void CreateTeacher()
        {
            List<RogueTeacher.Actions> actions = new List<RogueTeacher.Actions>();
            actions.Add(RogueTeacher.GetRandomAction(m_rndGen, (int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM]));

            Point teachersPoint;
            if ((int)TSHints[LTCopyAction.TEACHER_ON_DIFF_START_POSITION] != 0)
            {
                teachersPoint = World.RandomPositionInsidePow(m_rndGen, RogueTeacher.GetDefaultSize());
            }
            else
            {
                teachersPoint = new Point(m_agent.X + World.POW_WIDTH / 3, m_agent.Y);
            }

            m_teacher = (World as RoguelikeWorld).CreateTeacher(teachersPoint, actions) as RogueTeacher;
        }
    }
}
