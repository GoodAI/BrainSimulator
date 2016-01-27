using GoodAI.Modules.School.Common;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectShapeColor : DeprecatedAbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected Shape m_target;
        private MovableGameObject m_agent;

        public LTDetectShapeColor() { }

        public LTDetectShapeColor(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000 },
                { TSHintAttributes.VARIABLE_SIZE, 0 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.VARIABLE_SIZE, 0);
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();

            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                if (LearningTaskHelpers.FlipCoin(m_rndGen))
                {
                    CreateTarget(Shape.Shapes.Circle);
                }
                else
                {
                    CreateTarget(Shape.Shapes.Square);
                }
                SetTargetColor();
            }
            else
            {
                m_target = null;
            }
        }

        protected virtual void SetTargetColor()
        {
            m_target.isBitmapAsMask = true;
            LearningTaskHelpers.RandomizeColor(ref m_target.maskColor, m_rndGen);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            bool wasCircleTargetDetected = (World as ManInWorld).Controls.Host[0] != 0;
            bool wasSquareTargetDetected = (World as ManInWorld).Controls.Host[1] != 0;

            // both target detected
            if (wasCircleTargetDetected && wasSquareTargetDetected)
            {
                wasUnitSuccessful = false;
                GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
                return true;
            }

            // no target
            if (m_target == null)
            {
                if (wasSquareTargetDetected || wasCircleTargetDetected)
                {
                    wasUnitSuccessful = false;
                }
                else
                {
                    wasUnitSuccessful = true;
                }
                GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
                return true;
            }

            wasUnitSuccessful = (wasCircleTargetDetected && m_target.ShapeType == Shape.Shapes.Circle ||
                wasSquareTargetDetected && m_target.ShapeType == Shape.Shapes.Square);

            GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
            return true;
        }

        protected void CreateAgent()
        {
            World.CreateAgent(null, 0, 0);
            m_agent = World.Agent;
            // center the agent
            m_agent.X = World.FOW_WIDTH / 2 - m_agent.Width / 2;
            m_agent.Y = World.FOW_HEIGHT - m_agent.Height;
        }

        protected void CreateTarget(Shape.Shapes shape)
        {
            m_target = new Shape(shape, 0, 0);
            World.AddGameObject(m_target);
            m_target.X = m_rndGen.Next(m_agent.X - World.POW_WIDTH / 2, m_agent.X + World.POW_WIDTH / 2 - m_target.Width + 1);
            if (TSHints[TSHintAttributes.VARIABLE_SIZE] > 0)
            {
                double resizeRatio = m_rndGen.NextDouble() * 3 + 1.0d;
                m_target.Height = (int)(resizeRatio * m_target.Height);
                m_target.Width = (int)(resizeRatio * m_target.Width);
            }

            m_target.X = m_rndGen.Next(m_agent.X - World.POW_WIDTH / 2, m_agent.X + World.POW_WIDTH / 2 - m_target.Width + 1);
            m_target.Y = World.FOW_HEIGHT - m_target.Height;
        }

    }

    /*
        public class RoguelikeWorldWADetectShapeColor : AbstractWADetectShapeColor
        {
            private Worlds m_w;
            private MovableGameObject m_agent;

            private string GetShapeAddr(Shape.Shapes shape)
            {
                switch (shape)
                {
                    case Shape.Shapes.Circle:
                        return @"WhiteCircle50x50.png";
                    case Shape.Shapes.Square:
                        return @"White10x10.png";
                }
                throw new ArgumentException("Unknown shape");
            }

            protected override AbstractSchoolWorld World
            {
                get
                {
                    return m_w;
                }
            }

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

            protected override void CreateTarget(TrainingSetHints trainingSetHints, Shape.Shapes shape)
            {
                m_target = new Shape(shape, 0, 0);
                m_w.AddGameObject(m_target);

                if (trainingSetHints[TSHintAttributes.VARIABLE_SIZE] > 0)
                {
                    double resizeRatio = m_rndGen.NextDouble() * 3 + 1.0d;
                    m_target.Height = (int)(resizeRatio * m_target.Height);
                    m_target.Width = (int)(resizeRatio * m_target.Width);
                }

                m_target.X = m_rndGen.Next(0, m_w.POW_WIDTH - m_target.Width + 1);
                m_target.Y = m_rndGen.Next(0, m_w.POW_HEIGHT - m_target.Height + 1);
            }

            protected void CreateAgent()
            {
                m_w.CreateAgent(null, 0, 0);
                m_agent = m_w.Agent;
                // center the agent
                m_agent.X = m_w.POW_WIDTH / 2 - m_agent.Width / 2;
                m_agent.Y = m_w.POW_HEIGHT / 2 - m_agent.Height / 2;
            }
        }
     */
}
