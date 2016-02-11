using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTApproach : AbstractLearningTask<RoguelikeWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected GameObject m_agent;
        protected int m_stepsSincePresented = 0;
        protected float m_initialDistance = 0;

        public readonly TSHintAttribute DISTANCE_BONUS_COEFFICENT = new TSHintAttribute("Distant bonus coefficient", "", typeof(float), 0, 1);
        // DISTANCE_BONUS_COEFFICENT explanation: "return m_stepsSincePresented > m_initialDistance" is used to decide if the training unit failed, this means that
        // the unit fails unless the agent goes just to the right direction (towards the target) from the beginning.
        // DISTANCE_BONUS_COEFFICENT's default value is 1, and if it's 2 the amount of available steps to reach the target is doubled, new formula : "return m_stepsSincePresented > (m_initialDistance * (int)TSHints[MULTIPLY_COEFFICENT]);"


        public LTApproach() : base(null) { }

        public LTApproach(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
            };

            TSHints.Add(DISTANCE_BONUS_COEFFICENT, 1);

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.DegreesOfFreedom = (int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM];

            CreateAgent();
            CreateTarget();

            m_stepsSincePresented = 0;
            m_initialDistance = m_agent.DistanceTo(m_target);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            m_stepsSincePresented++;

            // TODO: partial reward

            if (DidTrainingUnitFail())
            {
                wasUnitSuccessful = false;
                return true;
            }

            float dist = m_agent.DistanceTo(m_target);
            if (dist < 7)
            {
                wasUnitSuccessful = true;
                return true;
            }
            wasUnitSuccessful = false;
            return false;
        }

        public virtual bool DidTrainingUnitFail()
        {
            return m_stepsSincePresented > (int)(m_initialDistance * (float)TSHints[DISTANCE_BONUS_COEFFICENT]);
        }

        protected void CreateAgent()
        {
            WrappedWorld.CreateAgent();
            m_agent = WrappedWorld.Agent;
            m_agent.X -= m_agent.Width / 2;
            m_agent.Y -= m_agent.Height / 2;
        }

        public virtual void CreateTarget()
        {
            float scaleFactor = 1;
            if(TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1){
                scaleFactor = (float)m_rndGen.NextDouble() * 0.7f + 0.8f;
            }

            m_target = WrappedWorld.CreateTarget(new Point(0, 0), scaleFactor);

            Point p;
            if ((int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] == 1)
            {
                Rectangle POW = WrappedWorld.GetPowGeometry();
                POW.Location = new Point(POW.X, POW.Y + POW.Height / 2 - m_agent.Height);
                POW.Size = new Size(POW.Width, m_agent.Height * 2);
                p = WrappedWorld.RandomPositionInsideRectangleNonCovering(m_rndGen, m_target.GetGeometry().Size, POW, 10);
            }
            else
            {
                p = WrappedWorld.RandomPositionInsidePow(m_rndGen, m_target.GetGeometry().Size, 20);
            }
            
            m_target.SetPosition(p);
        }

        private void PutAgentOnFloor()
        {
            m_agent.Y = WrappedWorld.FOW_HEIGHT - m_agent.Height - 1;  // - 1 : otherwise the agent is stuck in the floor
        }

    }
}
