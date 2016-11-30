using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Categorize object angles")]
    public class LTSimpleAngle : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly TSHintAttribute ERROR_TOLERANCE = new TSHintAttribute("Tolerance in rads", "", typeof(float), 0, 1); //check needed;
        private readonly TSHintAttribute FIXED_DISTANCE = new TSHintAttribute("Fixed distance to target", "", typeof(bool), 0, 1); //check needed;

        protected readonly Random m_rndGen = new Random();
        protected MovableGameObject m_agent;
        private GameObject m_target;

        public LTSimpleAngle() : this(null) { }

        public LTSimpleAngle(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 0 },
                { TSHintAttributes.IS_VARIABLE_COLOR, 0 },
                { TSHintAttributes.IMAGE_NOISE, 0},
                { ERROR_TOLERANCE , 0.2f},
                { FIXED_DISTANCE, 1},
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 1);
            TSProgression.Add(ERROR_TOLERANCE, 0.1f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(ERROR_TOLERANCE, 0.05f);
        }

        public override void PresentNewTrainingUnit()
        {
            m_agent = WrappedWorld.CreateNonVisibleAgent();

            SizeF size;
            if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1)
            {
                float side = (float)(8 + m_rndGen.NextDouble() * 16);
                size = new SizeF(side, side);
            }
            else
            {
                int side = 10;
                size = new Size(side, side);
            }

            PointF position;
            float radius = Math.Min(WrappedWorld.Viewport.Height, WrappedWorld.Viewport.Width) / 3;
            if (TSHints[FIXED_DISTANCE] >= 1)
            {
                double angle = m_rndGen.NextDouble() * Math.PI * 2;
                position = new PointF((float)(Math.Cos(angle) * radius), (float)(Math.Sin(angle) * radius));
                position += new SizeF(m_agent.Position.X, m_agent.Position.Y);
            }
            else
            {
                position = WrappedWorld.RandomPositionInsideViewport(m_rndGen, size);
            }

            Shape.Shapes shape;
            if (TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS] >= 1)
            {
                switch (m_rndGen.Next(0, 4))
                {
                    case 0:
                    default:
                        shape = Shape.Shapes.Circle;
                        break;
                    case 1:
                        shape = Shape.Shapes.Square;
                        break;
                    case 2:
                        shape = Shape.Shapes.Triangle;
                        break;
                    case 3:
                        shape = Shape.Shapes.Mountains;
                        break;
                }
            }
            else
            {
                shape = Shape.Shapes.Circle;
            }

            Color color;
            if (TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1)
            {
                color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
            }
            else
            {
                color = Color.White;
            }

            m_target = WrappedWorld.CreateShape(shape, color, position, size);
        }

        public static float EuclideanDistance(PointF p1, PointF p2)
        {
            float dx = p1.X - p2.X;
            float dy = p1.Y - p2.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        public static float RelativeSin(PointF point, PointF related)
        {
            return (point.Y - related.Y) / EuclideanDistance(point, related);
        }

        public static float RelativeCos(PointF point, PointF related)
        {
            return (point.X - related.X) / EuclideanDistance(point, related);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            float tolerance = TSHints[ERROR_TOLERANCE];

            float sin = RelativeSin(m_target.GetGeometry().Location, m_agent.GetGeometry().Location);
            float cos = RelativeCos(m_target.GetGeometry().Location, m_agent.GetGeometry().Location);
            //Console.WriteLine(sin);
            //Console.WriteLine(cos);
            sin = (sin + 1) / 2;
            cos = (cos + 1) / 2;
            //Console.WriteLine(sin);
            //Console.WriteLine(cos);
            if ((sin - tolerance <= WrappedWorld.Controls.Host[0] && WrappedWorld.Controls.Host[0] <= sin + tolerance) &&
                (cos - tolerance <= WrappedWorld.Controls.Host[1] && WrappedWorld.Controls.Host[1] <= cos + tolerance))
            {
                wasUnitSuccessful = true;
            }
            else
            {
                wasUnitSuccessful = false;
            }

            // TODO: Partial reward

            //Console.WriteLine(wasUnitSuccessful);
            return true;
        }
    }
}
