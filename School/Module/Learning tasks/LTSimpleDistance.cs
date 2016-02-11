using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleDistance : AbstractLearningTask<ManInWorld>
    {
        private readonly TSHintAttribute ERROR_TOLERANCE = new TSHintAttribute("Error tolerence in [0,1]", "", typeof(float), 0, 1);

        private Random m_rndGen = new Random();
        private GameObject m_agent;
        private GameObject m_target;
        private float m_distance = 0; // ranging from 0 to 1; 0-0.125 is smallest, 0.875-1 is biggest; m_distance is lower bound of the interval

        public LTSimpleDistance() : base(null) { }

        public LTSimpleDistance(SchoolWorld w)
            : base(w)
        {

            TSHints = new TrainingSetHints
            {
                { TSHintAttributes.IS_VARIABLE_COLOR, 0 },
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { ERROR_TOLERANCE, 0.25f },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.IS_VARIABLE_COLOR, 1 },
                { ERROR_TOLERANCE, 0.15f },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 0 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.IS_VARIABLE_SIZE, 1 },
                { ERROR_TOLERANCE, 0.10f },
            });
            TSProgression.Add(new TrainingSetHints {
                { ERROR_TOLERANCE, 0.05f },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3 },
            });
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
        }

        protected override void Init()
        {
            WrappedWorld.FreezeWorld(true);

            CreateAgent();
            CreateTarget();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            float tolerance = TSHints[ERROR_TOLERANCE];
            //Console.WriteLine(m_distance);
            //Console.WriteLine(m_distance - tolerance);
            //Console.WriteLine(m_distance + tolerance);
            // require immediate decision - in a single step
            if (m_distance - tolerance <= WrappedWorld.Controls.Host[0] && WrappedWorld.Controls.Host[0] <= m_distance + tolerance)
            {
                wasUnitSuccessful = true;
            }
            else
            {
                wasUnitSuccessful = false;
            }
            //Console.WriteLine(wasUnitSuccessful);
            // TODO: partial reward
            return true;
        }

        private void CreateAgent()
        {
            WrappedWorld.CreateAgent(null, 0, 0);
            m_agent = WrappedWorld.Agent;
            // center the agent
            m_agent.X = WrappedWorld.FOW_WIDTH / 2 - m_agent.Width / 2;
            m_agent.Y = WrappedWorld.FOW_HEIGHT / 2 - m_agent.Height / 2;
        }

        // scale and position the target:
        private void CreateTarget()
        {
            Size size;
            int standardSideSize = WrappedWorld.POW_WIDTH / 10;
            if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1)
            {
                int side = standardSideSize + m_rndGen.Next(standardSideSize);
                size = new Size(side, side);
            }
            else
            {
                size = new Size(standardSideSize, standardSideSize);
            }

            Point position = WrappedWorld.RandomPositionInsidePow(m_rndGen, size, -1);

            Shape.Shapes shape;
            switch (m_rndGen.Next(0, (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]))
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

            Color color;
            if (TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1)
            {
                color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
            }
            else
            {
                color = Color.White;
            }

            m_target = WrappedWorld.CreateShape(position, shape, color, size);

            float distance = m_target.CenterDistanceTo(m_agent);
            float maxDistance = (float)Math.Sqrt(Math.Pow(WrappedWorld.POW_WIDTH / 2, 2) + Math.Pow(WrappedWorld.POW_HEIGHT / 2, 2));
            m_distance = distance / maxDistance;
        }
    }
}
