
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectDifference : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected bool m_diffObjectetPlaced;

        public LTDetectDifference() { }

        public LTDetectDifference(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.IS_VARIABLE_COLOR, 0},
                {TSHintAttributes.IS_VARIABLE_SIZE, 0},
                {TSHintAttributes.NUMBER_OBJECTS, 2},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OBJECTS, 4f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1f);
            TSProgression.Add(TSHintAttributes.NUMBER_OBJECTS, 8f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1f);
            TSProgression.Add(TSHintAttributes.NUMBER_OBJECTS, 10f);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            if (WrappedWorld.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = WrappedWorld as RoguelikeWorld;

                world.CreateNonVisibleAgent();

                int numberOfShapes = Enum.GetValues(typeof(Shape.Shapes)).Length;
                List<int> uniqueCouple = LearningTaskHelpers.UniqueNumbers(m_rndGen, 0, numberOfShapes, 2);
                Shape.Shapes standardShape = (Shape.Shapes)uniqueCouple[0];
                Shape.Shapes alternativeShape = (Shape.Shapes)uniqueCouple[1];

                int numberOfObjects = (int)TSHints[TSHintAttributes.NUMBER_OBJECTS];

                m_diffObjectetPlaced = m_rndGen.Next(2) == 0 ? true : false;
                bool placeDifferentObj = m_diffObjectetPlaced;

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

                    Point position = world.RandomPositionInsidePowNonCovering(m_rndGen, size);

                    if (placeDifferentObj)
                    {
                        placeDifferentObj = false;
                        world.CreateShape(position, alternativeShape, color, size: size);
                    }
                    else
                    {
                        world.CreateShape(position, standardShape, color, size: size);
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
            if (m_diffObjectetPlaced == (WrappedWorld.Controls.Host[0] > 0))
            {
                wasUnitSuccessful = true;
            }
            return true;
        }

    }
}
