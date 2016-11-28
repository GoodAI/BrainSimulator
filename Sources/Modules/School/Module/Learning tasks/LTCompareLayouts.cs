using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Detect similar sets")]
    public class LTCompareLayouts : AbstractLearningTask<RoguelikeWorld>
    {
        protected readonly Random m_rndGen = new Random();
        protected bool m_diffObjectetPlaced;

        public LTCompareLayouts() : this(null) { }

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

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();

            RectangleF leftPart = WrappedWorld.GetPowGeometry();
            RectangleF rightPart = WrappedWorld.GetPowGeometry();
            SizeF shift = new SizeF(leftPart.Width / 2 + 4, 0);

            leftPart.Width = leftPart.Width / 2 - 4;
            rightPart.Width = rightPart.Width / 2 - 2;
            rightPart.X += rightPart.Width + 4;

            WrappedWorld.CreateShape(
                Shape.Shapes.Square,
                Color.Black,
                rightPart.Location - new SizeF(4, 0),
                new SizeF(4, leftPart.Height));

            int numberOfObjects = (int)TSHints[TSHintAttributes.NUMBER_OBJECTS];

            m_diffObjectetPlaced = m_rndGen.Next(2) == 0;

            for (int i = 0; i < numberOfObjects; i++)
            {
                SizeF size;
                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1f)
                {
                    float a = (float)(10 + m_rndGen.NextDouble() * 10);
                    size = new SizeF(a, a);
                }
                else
                {
                    size = new SizeF(15, 15);
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

                PointF position = WrappedWorld.RandomPositionInsideRectangleNonCovering(m_rndGen, size, leftPart, 2);

                bool placeDifference = m_diffObjectetPlaced;

                if (!placeDifference || i != numberOfObjects - 1)
                {
                    WrappedWorld.CreateShape(shape, color, position, size);
                    WrappedWorld.CreateShape(shape, color, position + shift, size);
                }
                else
                {
                    PointF positionR = position + shift;
                    Color colorR = color;
                    Shape.Shapes shapeR = shape;
                    SizeF sizeR = size;

                    if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1.0f)
                    {
                        positionR = WrappedWorld.RandomPositionInsideRectangleNonCovering(m_rndGen, size, rightPart, 2);
                        placeDifference = false;
                    }
                    if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1f)
                    {
                        colorR = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
                        placeDifference = false;
                    }
                    if (LearningTaskHelpers.FlipCoin(m_rndGen) && TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1f)
                    {
                        sizeR = size + new Size(5, 5);
                    }
                    if (placeDifference || LearningTaskHelpers.FlipCoin(m_rndGen))
                    {
                        shapeR = Shape.GetRandomShape(m_rndGen);
                    }

                    WrappedWorld.CreateShape(shape, color, position, size);
                    WrappedWorld.CreateShape(shapeR, colorR, positionR, sizeR);
                }
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
