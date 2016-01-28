
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTCompareLayouts : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected bool m_diffObjectetPlaced;

        public LTCompareLayouts() { }

        public LTCompareLayouts(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.RANDOMNESS_LEVEL, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, .3f); // different color
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, .6f); // different size
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.0f); // different amount
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            if (World.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = World as RoguelikeWorld;

                world.CreateNonVisibleAgent();

                Rectangle leftPart = World.GetPowGeometry();
                Rectangle rightPart = World.GetPowGeometry();
                Size shift = new Size(leftPart.Width / 2, 0);
                leftPart.Width = leftPart.Width / 2 - 20;
                rightPart.Width = rightPart.Width / 2 - 2;
                rightPart.X += rightPart.Width + 4;

                world.CreateShape(rightPart.Location - new Size(4, 0),
                    Shape.Shapes.Square,
                    Color.Black,
                    new Size(4, leftPart.Height));

                int numberOfObjects;
                if (TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= 1.0f)
                {
                    numberOfObjects = m_rndGen.Next(5, 10);
                }
                else
                {
                    numberOfObjects = 4;
                }

                m_diffObjectetPlaced = m_rndGen.Next(2) == 0 ? true : false;
                bool placeDifference = m_diffObjectetPlaced;

                for (int i = 0; i < numberOfObjects; i++)
                {

                    Size size;
                    if (TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= .6f)
                    {
                        int a = 10 + m_rndGen.Next(10);
                        size = new Size(a, a);
                    }
                    else
                    {
                        size = new Size(15, 15);
                    }

                    Color color;
                    if (TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= .3)
                    {
                        color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);

                    }
                    else
                    {
                        color = Color.White;
                    }

                    Shape.Shapes shape = Shape.GetRandomShape(m_rndGen);

                    Point position = world.RandomPositionInsideRectangleNonCovering(m_rndGen, size, leftPart);

                    if (placeDifference && i == numberOfObjects - 1)
                    {
                        placeDifference = false;

                        Point positionR = position + shift;
                        Color colorR = color;
                        Shape.Shapes shapeR = shape;
                        Size sizeR = size;

                        if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= 1.0f)
                        {
                            positionR = world.RandomPositionInsideRectangleNonCovering(m_rndGen, size, rightPart); placeDifference = false;
                        }
                        if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= .3)
                        {
                            colorR = LearningTaskHelpers.RandomVisibleColor(m_rndGen); placeDifference = false;
                        }
                        if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= .6f)
                        {
                            sizeR = size + new Size(5, 5);
                        }
                        if (placeDifference || LearningTaskHelpers.FlipCoin(m_rndGen))
                        {
                            shapeR = Shape.GetRandomShape(m_rndGen); placeDifference = false;
                        }
                        world.CreateShape(position, shape, color, size: size);
                        world.CreateShape(positionR, shapeR, colorR, size: sizeR);
                    }
                    else
                    {
                        world.CreateShape(position, shape, color, size: size);
                        world.CreateShape(position + shift, shape, color, size: size);
                    }
                }
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (m_diffObjectetPlaced != (World.Controls.Host[0] > 0))
            {
                wasUnitSuccessful = true;
            }
            return true;
        }

    }
}
