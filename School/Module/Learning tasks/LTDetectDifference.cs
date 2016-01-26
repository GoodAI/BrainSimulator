
using System;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System.Drawing;
using System.Collections.Generic;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectDifference : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected bool m_diffObjectetPlaced;

        public LTDetectDifference(ManInWorld w) : base (w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.NOISE, 0},
                {TSHintAttributes.RANDOMNESS, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS, .3f); // different color
            TSProgression.Add(TSHintAttributes.RANDOMNESS, .6f); // different size
            TSProgression.Add(TSHintAttributes.RANDOMNESS, 1.0f); // different amount
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            if (World.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = World as RoguelikeWorld;

                world.CreateNonVisibleAgent();

                int numberOfShapes = Enum.GetValues(typeof(Shape.Shapes)).Length;
                List<int> uniqueCouple = LearningTaskHelpers.UniqueNumbers(m_rndGen, 0, numberOfShapes, 2);
                Shape.Shapes standardShape = (Shape.Shapes)uniqueCouple[0];
                Shape.Shapes alternativeShape = (Shape.Shapes)uniqueCouple[1];

                int numberOfObjects;
                if (TSHints[TSHintAttributes.RANDOMNESS] >= 1.0f)
                {
                    numberOfObjects = 5 + m_rndGen.Next(5);
                }
                else
                {
                    numberOfObjects = 7;
                }

                m_diffObjectetPlaced = m_rndGen.Next(2) == 0 ? true : false;
                bool placeDifferentObj = m_diffObjectetPlaced;

                for (int i = 0; i < numberOfObjects; i++)
                {

                    Size s;
                    if(TSHints[TSHintAttributes.RANDOMNESS] >= .6f)
                    {
                        int a = 10 + m_rndGen.Next(10);
                        s = new Size(a, a);
                    }
                    else
                    {
                        s = new Size(15,15);
                    }

                    Color color;
                    if (TSHints[TSHintAttributes.RANDOMNESS] >= .3)
                    {
                        color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
                    }
                    else
                    {
                        color = Color.White;
                    }

                    Point position = world.GetRandomPositionInsidePowNonCovering(m_rndGen, s);

                    if (placeDifferentObj)
                    {
                        placeDifferentObj = false;
                        world.CreateShape(position, alternativeShape, color, size : s);
                    } 
                    else
                    {
                        world.CreateShape(position, standardShape, color, size: s);
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
            if (m_diffObjectetPlaced == (World.Controls.Host[0] > 0))
            {
                wasUnitSuccessful = true;
            }
            return true;
        }

    }
}
