
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTClassComposition : AbstractLearningTask<ManInWorld>
    {
        private static readonly TSHintAttribute CARDINALITY_OF_SET = new TSHintAttribute("Cardinality of set","",TypeCode.Single,0,1); //check needed;
        private static readonly TSHintAttribute IS_TARGET_MOVING = new TSHintAttribute("Is target moving","",TypeCode.Single,0,1); //check needed;

        protected Random m_rndGen = new Random();
        protected bool m_positiveExamplePlaced;
        //private int m_numberOfObjects;
        private List<Shape.Shapes> m_positiveExamples;
        private List<Shape.Shapes> m_negativeExamples;

        public LTClassComposition() { }

        public LTClassComposition(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.RANDOMNESS_LEVEL, 0},
                {CARDINALITY_OF_SET, 2},
                {IS_TARGET_MOVING, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(IS_TARGET_MOVING, 1);;
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 0.3f);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 0.6f);
            TSProgression.Add(CARDINALITY_OF_SET, 3);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.0f);
            TSProgression.Add(CARDINALITY_OF_SET, 5); // max
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);

            int numberOfObjects = (int) TSHints[CARDINALITY_OF_SET];

            List<Shape.Shapes> positiveExamples = new List<Shape.Shapes>();
            positiveExamples.Add(Shape.Shapes.Star);
            positiveExamples.Add(Shape.Shapes.Circle);
            positiveExamples.Add(Shape.Shapes.T);
            positiveExamples.Add(Shape.Shapes.Tent);
            positiveExamples.Add(Shape.Shapes.Mountains);
            List<Shape.Shapes> negativeExamples = new List<Shape.Shapes>();
            negativeExamples.Add(Shape.Shapes.DoubleRhombus);
            negativeExamples.Add(Shape.Shapes.Pentagon);
            negativeExamples.Add(Shape.Shapes.Rhombus);
            negativeExamples.Add(Shape.Shapes.Square);
            negativeExamples.Add(Shape.Shapes.Triangle);

            m_positiveExamples = new List<Shape.Shapes>();
            m_negativeExamples = new List<Shape.Shapes>();
            for (int i = 0; i < numberOfObjects; i++)
            {
                m_positiveExamples.Add(positiveExamples[i]);
                m_negativeExamples.Add(negativeExamples[i]);
            }
            
        }

        protected override void PresentNewTrainingUnit()
        {
            if (World.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = World as RoguelikeWorld;

                world.CreateNonVisibleAgent();

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

                Point position;
                if (TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= 1.0f)
                {
                    position = world.RandomPositionInsidePow(m_rndGen, size);
                }
                else
                {
                    position = world.Agent.GetGeometry().Location;
                }

                m_positiveExamplePlaced = LearningTaskHelpers.FlipCoin(m_rndGen);

                Shape.Shapes shape;
                if (m_positiveExamplePlaced)
                {
                    int randShapePointer = m_rndGen.Next(0, m_positiveExamples.Count);
                    shape = m_positiveExamples[randShapePointer];
                }
                else
                {
                    int randShapePointer = m_rndGen.Next(0, m_negativeExamples.Count);
                    shape = m_negativeExamples[randShapePointer];
                }

                World.CreateShape(position, shape, color, size);
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (m_positiveExamplePlaced == (World.Controls.Host[0] != 0))
            {
                wasUnitSuccessful = true;
            }
            else
            {
                wasUnitSuccessful = false;
            }
            return true;
        }

    }
}
