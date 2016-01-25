
using System;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectSimilarity : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected bool m_sameObjectetPlaced;

        public LTDetectSimilarity(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.NOISE, 1},
                {TSHintAttributes.RANDOMNESS, 1},
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
            World.ClearWorld();
            if (World.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = World as RoguelikeWorld;

                world.CreateNonVisibleAgent();



                int numberOfObjects;
                if (TSHints[TSHintAttributes.RANDOMNESS] > 1.0f)
                {
                    numberOfObjects = m_rndGen.Next(5, 10);
                }
                else
                {
                    numberOfObjects = 7;
                }

                m_sameObjectetPlaced = m_rndGen.Next(2) == 0 ? true : false;

                if (m_sameObjectetPlaced)
                {
                    numberOfObjects--;
                }

                int numberOfShapes = Enum.GetValues(typeof(Shape.Shapes)).Length;
                List<int> uniqueNumbers = LearningTaskHelpers.UniqueNumbers(m_rndGen, 0, numberOfShapes, numberOfObjects);
                List<Shape.Shapes> shapes = uniqueNumbers.Select(x => (Shape.Shapes)x).ToList();
                if (m_sameObjectetPlaced)
                {
                    shapes.Add(shapes.Last());
                }
                
                for (int i = 0; i < shapes.Count; i++)
                {

                    Size s;
                    if (TSHints[TSHintAttributes.RANDOMNESS] > .6f)
                    {
                        int a = 10 + m_rndGen.Next(10);
                        s = new Size(a, a);
                    }
                    else
                    {
                        s = new Size(15, 15);
                    }

                    Color color;
                    if (TSHints[TSHintAttributes.RANDOMNESS] > .3)
                    {
                        color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
                    }
                    else
                    {
                        color = Color.White;
                    }


                    Point position = world.GetRandomPositionInsidePowNonCovering(m_rndGen, s);

                    world.CreateShape(position, shapes[i], color, size: s);
                }
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (m_sameObjectetPlaced == (World.Controls.Host[0] > 0))
            {
                wasUnitSuccessful = true;
            }
            return true;
        }

    }
}
