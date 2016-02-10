
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

        public LTCompareLayouts() : base(null) { }

        public LTCompareLayouts(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.IS_VARIABLE_COLOR, 0},
                {TSHintAttributes.IS_VARIABLE_SIZE, 0},
                {TSHintAttributes.IS_VARIABLE_POSITION, 0},
                {TSHintAttributes.NUMBER_OBJECTS, 2},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OBJECTS, 4f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1f);
            TSProgression.Add(TSHintAttributes.NUMBER_OBJECTS, 8f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1f);
            TSProgression.Add(TSHintAttributes.NUMBER_OBJECTS, 10f);
        }

        protected override void PresentNewTrainingUnit()
        {
            if (WrappedWorld.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = WrappedWorld as RoguelikeWorld;

                world.CreateNonVisibleAgent();

                Rectangle leftPart = WrappedWorld.GetPowGeometry();
                Rectangle rightPart = WrappedWorld.GetPowGeometry();
                Size shift = new Size(leftPart.Width / 2 + 4, 0);

                leftPart.Width = leftPart.Width / 2 - 4;
                rightPart.Width = rightPart.Width / 2 - 2;
                rightPart.X += rightPart.Width + 4;

                world.CreateShape(rightPart.Location - new Size(4, 0),
                    Shape.Shapes.Square,
                    Color.Black,
                    new Size(4, leftPart.Height));

                int numberOfObjects = (int)TSHints[TSHintAttributes.NUMBER_OBJECTS];

                m_diffObjectetPlaced = m_rndGen.Next(2) == 0 ? true : false;
                bool placeDifference = m_diffObjectetPlaced;

                for (int i = 0; i < numberOfObjects; i++)
                {

                    Size size;
                    if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1f)
                    {
                        int a = 10 + m_rndGen.Next(10);
                        size = new Size(a, a);
                    }
                    else
                    {
                        size = new Size(15, 15);
                    }

                    Color color;
                    if (TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1f)
                    {
                        color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);

                    }
                    else
                    {
                        color = Color.White;
                    }

                    Shape.Shapes shape = Shape.GetRandomShape(m_rndGen);

                    Point position = world.RandomPositionInsideRectangleNonCovering(m_rndGen, size, leftPart, 2);

                    if (placeDifference && i == numberOfObjects - 1)
                    {
                        placeDifference = false;

                        Point positionR = position + shift;
                        Color colorR = color;
                        Shape.Shapes shapeR = shape;
                        Size sizeR = size;

                        if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1.0f)
                        {
                            positionR = world.RandomPositionInsideRectangleNonCovering(m_rndGen, size, rightPart, 2); placeDifference = false;
                        }
                        if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1f)
                        {
                            colorR = LearningTaskHelpers.RandomVisibleColor(m_rndGen); placeDifference = false;
                        }
                        if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1f)
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
            if (m_diffObjectetPlaced != (WrappedWorld.Controls.Host[0] > 0))
            {
                wasUnitSuccessful = true;
            }
            return true;
        }
    }
}
