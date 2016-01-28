using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LT1DApproach : DeprecatedAbstractLearningTask<ManInWorld>
    {
        public LT1DApproach() { }

        public LT1DApproach(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1},
                { TSHintAttributes.NOISE, 0},
                { TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 0f},
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 0f},
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1},
                { TSHintAttributes.MAX_TARGET_DISTANCE, 0f},
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000},
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 1f },
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 2f },
                { TSHintAttributes.NOISE, .5f },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 0f },
                { TSHintAttributes.MAX_TARGET_DISTANCE, .3f },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 1.5f },
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 3f },
                { TSHintAttributes.NOISE, 1f },
                { TSHintAttributes.MAX_TARGET_DISTANCE, -1f },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100 }
            });

            SetHints(TSHints);
        }

        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected GameObject m_agent;
        protected int m_stepsSincePresented = 0;
        protected float m_initialDistance = 0;

        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();
            CreateTarget();

            m_stepsSincePresented = 0;
            m_initialDistance = m_agent.DistanceTo(m_target);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;

            if (DidTrainingUnitFail())
            {
                wasUnitSuccessful = false;
                return true;
            }

            float dist = m_agent.DistanceTo(m_target);
            return wasUnitSuccessful = dist < 15;
        }

        // following is PlumberWorld specific TODO use inheritance or something else...

        protected bool DidTrainingUnitFail()
        {
            return m_stepsSincePresented > m_initialDistance;
        }

        protected void CreateAgent()
        {
            World.CreateAgent(@"Plumber24x28.png", 0, 0);
            m_agent = World.Agent;
            // center the agent
            m_agent.X = World.FOW_WIDTH / 2 - m_agent.Width / 2;
            m_agent.Y = World.FOW_HEIGHT - m_agent.Height - 1;  // - 1 : otherwise the agent is stuck in the floor
        }

        protected void CreateTarget()
        {
            m_target = new GameObject(GameObjectType.None, @"Block60x10.png", 0, 0);

            World.AddGameObject(m_target);

            // first implementation, random object position (left or right)
            int maxTargetDistance
                = TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] < 0 ? World.FOW_WIDTH : (int)(World.FOW_WIDTH * TSHints[TSHintAttributes.MAX_TARGET_DISTANCE]);

            bool isLeft = m_rndGen.Next(0, 2) == 1;
            int targetDistX = m_rndGen.Next(20, maxTargetDistance / 2 - m_agent.Width / 2 - m_target.Width / 2);
            int targetX = m_agent.X;
            if (isLeft)
                targetX -= (targetDistX + m_target.Width);
            else
                targetX += (targetDistX + m_agent.Width);
            m_target.X = targetX;
            m_target.Y = m_agent.Y;
        }
    }

    /* TODO add Rogueliekeworld implementation
    public class RoguelikeWorldWA1DApproach : AbstractWA1DApproach
    {

        protected override bool DidTrainingUnitFail()
        {
            return m_stepsSincePresented > 2 * m_initialDistance;
        }

        protected override void InstallWorld(AbstractSchoolWorld w, TrainingSetHints trainingSetHints)
        {
            World = w as RoguelikeWorld;
            m_w.ClearWorld();
            m_w.DegreesOfFreedom = (int)trainingSetHints[TSHintAttributes.DEGREES_OF_FREEDOM];
            if (trainingSetHints[TSHintAttributes.NOISE] > 0)
                m_w.IsImageNoise = true;
        }

        protected override void CreateAgent()
        {
            m_w.CreateAgent(@"Agent.png", m_w.FOW_WIDTH / 2, 50);
            m_agent = m_w.Agent;
            //m_agent.X -= m_agent.Width / 2;
            //m_agent.Y -= m_agent.Height / 2;
        }

        protected override void CreateTarget(TrainingSetHints trainingSetHints)
        {
            m_target = new GameObject(GameObjectType.None, @"Target2.png", 0, 0);
            m_w.AddGameObject(m_target);

            // first implementation, random object position (left or right)
            bool isLeft = m_rndGen.Next(0, 2) == 1;
            bool isUp = m_rndGen.Next(0, 2) == 1;

            const int MIN_TARGET_DISTANCE = 20;
            int maxTargetDistanceX = m_w.FOW_WIDTH / 2 - m_agent.Width / 2 - m_target.Height / 2;
            if (trainingSetHints[TSHintAttributes.MAX_TARGET_DISTANCE] < 0)
                maxTargetDistanceX = (int)(MIN_TARGET_DISTANCE + (maxTargetDistanceX - MIN_TARGET_DISTANCE) * trainingSetHints[TSHintAttributes.MAX_TARGET_DISTANCE]);

            int targetDistX = m_rndGen.Next(MIN_TARGET_DISTANCE, maxTargetDistanceX);
            int targetX = m_agent.X;
            if (isLeft)
                targetX -= targetDistX;
            else
                targetX += targetDistX;

            int targetY = m_agent.Y;

            m_target.X = targetX;
            m_target.Y = m_agent.Y;
        }
    }*/
}
