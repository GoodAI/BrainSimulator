using GoodAI.Modules.School.Common;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleAngle : DeprecatedAbstractLearningTask<ManInWorld>
    {
        public const string TOLERANCE = "Tolerance in rads";
        public const string FIXED_DISTANCE = "Fixed distance to target";

        protected Random m_rndGen = new Random();
        protected Shape m_target;
        protected MovableGameObject m_agent;

        public LTSimpleAngle() { }

        public LTSimpleAngle(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.VARIABLE_SIZE, 0 },
                { TSHintAttributes.NOISE, 0},
                { TOLERANCE , 0.2f},
                { FIXED_DISTANCE, 0},
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(FIXED_DISTANCE, 1);
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TOLERANCE, 0.1f);
            TSProgression.Add(TSHintAttributes.VARIABLE_SIZE, 1);
            TSProgression.Add(TOLERANCE, 0.05f);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();
            CreateTarget();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            double estimatedAngle = (World as ManInWorld).Controls.Host[0];

            double targetAngle = Math.Atan2(m_target.Y - m_agent.Y, m_target.X - m_agent.X) / 2 / Math.PI;
            if (targetAngle < 0) targetAngle = 1 + targetAngle;


            double upperBound = targetAngle + TSHints[LTSimpleAngle.TOLERANCE];
            if (upperBound > 1) upperBound -= 1;
            double lowerBound = targetAngle - TSHints[LTSimpleAngle.TOLERANCE];
            if (lowerBound < 0) upperBound += 1;

            wasUnitSuccessful = (estimatedAngle > lowerBound) && (estimatedAngle < upperBound);

            GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
            return true;
        }

        protected void CreateTarget()
        {
            m_target = new Shape(Shape.Shapes.Circle, 0, 0);
            World.AddGameObject(m_target);
            m_target.X = m_rndGen.Next(m_agent.X - World.POW_WIDTH / 2, m_agent.X + World.POW_WIDTH / 2 - m_target.Width + 1);

            if (TSHints[TSHintAttributes.VARIABLE_SIZE] > 0)
            {
                double resizeRatio = m_rndGen.NextDouble() * 3 + 1.0d;
                m_target.Height = (int)(resizeRatio * m_target.Height);
                m_target.Width = (int)(resizeRatio * m_target.Width);
            }

            double rad = m_rndGen.NextDouble() * Math.PI * 2;
            double distanceMultiplicator;
            if (TSHints[LTSimpleAngle.FIXED_DISTANCE] > 0)
            {
                distanceMultiplicator = 30;
            }
            else
            {
                distanceMultiplicator = 10 + m_rndGen.NextDouble() * 50;
            }
            double X = Math.Sin(rad) * distanceMultiplicator;
            double Y = Math.Cos(rad) * distanceMultiplicator;

            m_target.X = m_agent.X + (int)X;
            m_target.Y = m_agent.Y + (int)Y;
        }

        protected void CreateAgent()
        {
            World.CreateAgent(null, 0, 0);
            m_agent = World.Agent;
            // center the agent
            m_agent.X = World.FOW_WIDTH / 2 - m_agent.Width / 2;
            m_agent.Y = World.FOW_HEIGHT / 2 - m_agent.Height / 2;
        }
    }

    /*
        public class RoguelikeWorldWASimpleAngle : AbstractWASimpleAngle
        {
            protected override void InstallWorld(AbstractSchoolWorld w, TrainingSetHints trainingSetHints)
            {
                m_w = w as RoguelikeWorld;
                m_w.ClearWorld();
                if (trainingSetHints[TSHintAttributes.NOISE] > 0)
                {
                    m_w.IsImageNoise = true;
                }
                CreateAgent();
            }

            protected override void CreateTarget(TrainingSetHints trainingSetHints)
            {
                m_target = new Shape(Shape.Shapes.Circle, 0, 0);
                m_w.AddGameObject(m_target);
                m_target.X = m_rndGen.Next(m_agent.X - m_w.POW_WIDTH / 2, m_agent.X + m_w.POW_WIDTH / 2 - m_target.Width + 1);

                if (trainingSetHints[TSHintAttributes.VARIABLE_SIZE] > 0)
                {
                    double resizeRatio = m_rndGen.NextDouble() * 3 + 1.0d;
                    m_target.Height = (int)(resizeRatio * m_target.Height);
                    m_target.Width = (int)(resizeRatio * m_target.Width);
                }

                double rad = m_rndGen.NextDouble() * Math.PI * 2;
                double distanceMultiplicator;
                if (trainingSetHints[LTSimpleAngle.FIXED_DISTANCE] > 0)
                {
                    distanceMultiplicator = 10;
                }
                else
                {
                    distanceMultiplicator = 5 + m_rndGen.NextDouble() * 10;
                }
                double X = Math.Sin(rad) * distanceMultiplicator;
                double Y = Math.Cos(rad) * distanceMultiplicator;

                m_target.X = m_agent.X + (int)X;
                m_target.Y = m_agent.Y + (int)Y;
            }


            protected void CreateAgent()
            {
                m_w.CreateAgent(null, 0, 0);
                m_agent = m_w.Agent;
                // center the agent
                m_agent.X = m_w.FOW_WIDTH / 2 - m_agent.Width / 2;
                m_agent.Y = m_w.FOW_HEIGHT / 2 - m_agent.Height / 2;
            }
        }
     */
}
