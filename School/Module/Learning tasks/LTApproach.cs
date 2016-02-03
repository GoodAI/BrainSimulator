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

        public readonly TSHintAttribute DISTANCE_BONUS_COEFFICENT = new TSHintAttribute("Multiply coefficent", "", TypeCode.Single, 0, 1);
        // DISTANCE_BONUS_COEFFICENT explanation: "return m_stepsSincePresented > m_initialDistance" is used to decide if the training unit failed, this means that
        // the unit fails unless the agent goes just to the right direction (towards the target) from the beginning.
        // DISTANCE_BONUS_COEFFICENT's default value is 1, and if it's 2 the amount of available steps to reach the target is doubled, new formula : "return m_stepsSincePresented > (m_initialDistance * (int)TSHints[MULTIPLY_COEFFICENT]);"


        public LTApproach() { }

        public LTApproach(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 0 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE, .3f }
            };

            TSHints.Add(DISTANCE_BONUS_COEFFICENT, 1);

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);
            TSProgression.Add(TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE, -1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 1);
            TSProgression.Add(TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 1.5f);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();
            CreateTarget();
            AdjustTargetSize();

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
            if (dist < 15)
            {
                wasUnitSuccessful = true;
                return true;
            }
            wasUnitSuccessful = false;
            return false;
        }

        // Randomly shrink or expand target
        protected void AdjustTargetSize()
        {
            if (TSHints[TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION] != 0)
            {
                float scalingFactor = (float)Math.Pow(2, TSHints[TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION] * GetRandomGaussian());

                int oldTargetWidth = m_target.Width;
                m_target.Width = (int)(scalingFactor * m_target.Width);
                m_target.X -= (m_target.Width - oldTargetWidth) / 2;

                int oldTargetHeight = m_target.Height;
                m_target.Height = (int)(scalingFactor * m_target.Height);
                m_target.Y -= (m_target.Height - oldTargetHeight) / 2;
            }

            //PutAgentOnFloor();
        }

        /*
        protected void AdjustTargetSize(TrainingSetHints trainingSetHints)
        {
            if (trainingSetHints[TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION] != 0)
            {
                //base.AdjustTargetSize(trainingSetHints);
                PutAgentOnFloor();
            }
        }
*/

        protected float GetRandomGaussian()
        {
            float u1 = Convert.ToSingle(m_rndGen.NextDouble()); //these are uniform(0,1) random doubles
            float u2 = Convert.ToSingle(m_rndGen.NextDouble()); //these are uniform(0,1) random doubles
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2)); //random normal(0,1)
            return randStdNormal;
        }

        protected virtual string GetTargetImage(int numberOfAlternatives)
        {
            switch (m_rndGen.Next(0, numberOfAlternatives))
            {
                case 0:
                    //return "Block60x10.png"; // used to be GetDefaultTargetImage();
                    return "Target_TOP.png";
                case 1:
                    return "White10x10.png";
                case 2:
                default:
                    return "WhiteCircle50x50.png";
            }
        }

        public virtual bool DidTrainingUnitFail()
        {
            return m_stepsSincePresented > (int)(m_initialDistance * (float)TSHints[DISTANCE_BONUS_COEFFICENT]);
        }

        protected void CreateAgent()
        {
            // Plumber:
            //World.CreateAgent(@"Plumber24x28.png", 0, 0);
            //m_agent = World.Agent;
            //// center the agent
            //m_agent.X = World.FOW_WIDTH / 2 - m_agent.Width / 2;
            //PutAgentOnFloor();
            WrappedWorld.CreateAgent();
            m_agent = WrappedWorld.Agent;
            m_agent.X -= m_agent.Width / 2;
            m_agent.Y -= m_agent.Height / 2;
        }

        public virtual void CreateTarget()
        {
            // Plumber:
            //m_target = new GameObject(GameObjectType.None, GetTargetImage((int)TSHints[TSHintAttributes.TARGET_IMAGE_VARIABILITY]), 0, 0);

            //World.AddGameObject(m_target);

            //// first implementation, random object position (left or right)
            //int maxTargetDistance
            //    = TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] == -1 ? World.FOW_WIDTH : (int)(World.FOW_WIDTH * TSHints[TSHintAttributes.MAX_TARGET_DISTANCE]);

            //bool isLeft = m_rndGen.Next(0, 2) == 1;
            //int targetDistX = m_rndGen.Next(20, maxTargetDistance / 2 - m_agent.Width / 2 - m_target.Width / 2);
            //int targetX = m_agent.X;
            //if (isLeft)
            //    targetX -= (targetDistX + m_target.Width);
            //else
            //    targetX += (targetDistX + m_agent.Width);
            //m_target.X = targetX;
            //m_target.Y = World.FOW_HEIGHT - m_target.Height;

            m_target = WrappedWorld.CreateTarget(new Point(0,0));

            // TODO: hint resolving
            Point p = WrappedWorld.RandomPositionInsidePow(m_rndGen, m_target.GetGeometry().Size);
            m_target.SetPosition(p);

            // first implementation, random object position (left or right)
            //bool isLeft = m_rndGen.Next(0, 2) == 1;
            //bool isUp = m_rndGen.Next(0, 2) == 1;

            //const int MIN_TARGET_DISTANCE = 20;
            //int maxTargetDistanceX = WrappedWorld.FOW_WIDTH / 2 - m_agent.Width / 2 - m_target.Height / 2;
            //if (TSHints[TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE] != -1)
            //{
            //    maxTargetDistanceX = (int)(MIN_TARGET_DISTANCE + (maxTargetDistanceX - MIN_TARGET_DISTANCE) * TSHints[TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE]);
            //}
            //int targetDistX = m_rndGen.Next(MIN_TARGET_DISTANCE, maxTargetDistanceX);
            //int targetX = m_agent.X;
            //if (isLeft)
            //    targetX -= targetDistX;
            //else
            //    targetX += targetDistX;

            //int targetY = m_agent.Y;
            //switch ((int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM])
            //{
            //    case 1:
            //        // Center the target on the agent
            //        targetY += (m_agent.Height - m_target.Height) / 2;
            //        break;
            //    case 2:
            //        int maxTargetDistanceY = WrappedWorld.FOW_HEIGHT / 2 - m_agent.Height / 2 - m_target.Height / 2;
            //        if (TSHints[TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE] != -1)
            //        {
            //            maxTargetDistanceY = (int)(MIN_TARGET_DISTANCE + (maxTargetDistanceY - MIN_TARGET_DISTANCE) * TSHints[TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE]);
            //        }
            //        int targetDistY = m_rndGen.Next(MIN_TARGET_DISTANCE, maxTargetDistanceY);
            //        if (isUp)
            //            targetY -= targetDistY;
            //        else
            //            targetY += targetDistY;
            //        break;
            //}

            //m_target.X = targetX;
            //m_target.Y = targetY;
        }

        private void PutAgentOnFloor()
        {
            m_agent.Y = WrappedWorld.FOW_HEIGHT - m_agent.Height - 1;  // - 1 : otherwise the agent is stuck in the floor
        }

    }


    /*
        public class RoguelikeWorldWAApproach : AbstractWAApproach
        {
            private Worlds m_w;

            protected override AbstractSchoolWorld World
            {
                get
                {
                    return m_w;
                }
            }

            protected override bool DidTrainingUnitFail()
            {
                return m_stepsSincePresented > 2 * m_initialDistance;
            }

            protected override void InstallWorld(AbstractSchoolWorld w, TrainingSetHints trainingSetHints)
            {
                m_w = w as RoguelikeWorld;
                m_w.ClearWorld();
                m_w.DegreesOfFreedom = (int)trainingSetHints[TSHintAttributes.DEGREES_OF_FREEDOM];
                if (trainingSetHints[TSHintAttributes.NOISE] > 0)
                {
                    m_w.IsImageNoise = true;
                }
            }

            protected override void CreateAgent()
            {
                m_w.CreateAgent(@"Agent.png", m_w.FOW_WIDTH / 2, m_w.FOW_HEIGHT / 2);
                m_agent = m_w.Agent;
                m_agent.X -= m_agent.Width / 2;
                m_agent.Y -= m_agent.Height / 2;
            }

            protected override void CreateTarget(TrainingSetHints trainingSetHints)
            {
                m_target = new GameObject(GameObjectType.None, GetTargetImage((int)trainingSetHints[TSHintAttributes.TARGET_IMAGE_VARIABILITY]), 0, 0);
                m_w.AddGameObject(m_target);

                // first implementation, random object position (left or right)
                bool isLeft = m_rndGen.Next(0, 2) == 1;
                bool isUp = m_rndGen.Next(0, 2) == 1;

                const int MIN_TARGET_DISTANCE = 20;
                int maxTargetDistanceX = m_w.FOW_WIDTH / 2 - m_agent.Width / 2 - m_target.Height / 2;
                if (trainingSetHints[TSHintAttributes.MAX_TARGET_DISTANCE] != -1)
                {
                    maxTargetDistanceX = (int)(MIN_TARGET_DISTANCE + (maxTargetDistanceX - MIN_TARGET_DISTANCE) * trainingSetHints[TSHintAttributes.MAX_TARGET_DISTANCE]);
                }
                int targetDistX = m_rndGen.Next(MIN_TARGET_DISTANCE, maxTargetDistanceX);
                int targetX = m_agent.X;
                if (isLeft)
                    targetX -= targetDistX;
                else
                    targetX += targetDistX;

                int targetY = m_agent.Y;
                switch ((int)trainingSetHints[TSHintAttributes.DEGREES_OF_FREEDOM])
                {
                    case 1:
                        // Center the target on the agent
                        targetY += (m_agent.Height - m_target.Height) / 2;
                        break;
                    case 2:
                        int maxTargetDistanceY = m_w.FOW_HEIGHT / 2 - m_agent.Height / 2 - m_target.Height / 2;
                        if (trainingSetHints[TSHintAttributes.MAX_TARGET_DISTANCE] != -1)
                        {
                            maxTargetDistanceY = (int)(MIN_TARGET_DISTANCE + (maxTargetDistanceY - MIN_TARGET_DISTANCE) * trainingSetHints[TSHintAttributes.MAX_TARGET_DISTANCE]);
                        }
                        int targetDistY = m_rndGen.Next(MIN_TARGET_DISTANCE, maxTargetDistanceY);
                        if (isUp)
                            targetY -= targetDistY;
                        else
                            targetY += targetDistY;
                        break;
                }

                m_target.X = targetX;
                m_target.Y = targetY;
            }

            protected override string GetDefaultTargetImage()
            {
                return @"Target2.png";
            }
        }
     */
}
