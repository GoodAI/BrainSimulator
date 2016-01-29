using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleSize : AbstractLearningTask<ManInWorld>
    {
        private static readonly TSHintAttribute COLOR_PATTERNS = new TSHintAttribute("Color patterns","",TypeCode.Single,0,1); //check needed;
        private static readonly TSHintAttribute TARGET_SIZE_LEVELS = new TSHintAttribute("Target size levels","",TypeCode.Single,0,1); //check needed;

        private Random m_rndGen = new Random();
        private GameObject m_agent;
        private float m_scale;

        public LTSimpleSize() { }

        public LTSimpleSize(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { COLOR_PATTERNS, 0 },
                { TARGET_SIZE_LEVELS, 2 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(new TrainingSetHints {
                { COLOR_PATTERNS, 1 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 0 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TARGET_SIZE_LEVELS, 5 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TARGET_SIZE_LEVELS, 10 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3 },
            });
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            World.FreezeWorld(true);

            World.IsImageNoise = TSHints[TSHintAttributes.IMAGE_NOISE] >= 1 ? true : false;

            CreateAgent();
            CreateTarget();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // require immediate decision - in a single step
            float tolerance = 1.0f / (TSHints[LTSimpleSize.TARGET_SIZE_LEVELS] + 1);
            if (m_scale - tolerance <= World.Controls.Host[0] && World.Controls.Host[0] <= m_scale + tolerance)
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

        protected void CreateAgent()
        {
            m_agent = World.CreateNonVisibleAgent();
        }

        // scale and position the target:
        protected void CreateTarget()
        {
            // the number of different sizes depends on level:
            int maxWidth = (int)(World.POW_WIDTH * 0.9);
            int maxHeight = (int)(World.POW_HEIGHT * 0.9);
            float fRatio = m_rndGen.Next(1, (int)TSHints[LTSimpleSize.TARGET_SIZE_LEVELS] + 1) / (float)TSHints[LTSimpleSize.TARGET_SIZE_LEVELS];

            int maxSide = (int)(Math.Max(World.POW_WIDTH, World.POW_HEIGHT) * 0.9);

            float randomNumber = (float)(m_rndGen.Next(1, (int)TSHints[LTSimpleSize.TARGET_SIZE_LEVELS] + 1));
            m_scale = randomNumber / TSHints[LTSimpleSize.TARGET_SIZE_LEVELS];
            int side = (int)((float)maxSide * m_scale);

            //Console.WriteLine(maxSide);
            //Console.WriteLine(side);

            Size size = new Size(side, side);

            Point position = World.RandomPositionInsidePow(m_rndGen, size, true);

            List<Shape.Shapes> shapes = new List<Shape.Shapes>();
            switch ((int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS])
            {
                case 3:
                    shapes.Add(Shape.Shapes.Star);
                    goto case 2;
                case 2:
                    shapes.Add(Shape.Shapes.DoubleRhombus);
                    goto case 1;
                case 1:
                    shapes.Add(Shape.Shapes.Square);
                    break;
            }
            Shape.Shapes shape = shapes[m_rndGen.Next(0, (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS])];

            Color color;
            if (TSHints[LTSimpleSize.COLOR_PATTERNS] >= 1)
            {
                color = LearningTaskHelpers.FlipCoin(m_rndGen) ? Color.Black : Color.White;
            }
            else
            {
                color = Color.White;
            }

            World.CreateShape(position, shape, color, size);
        }

    }

}
