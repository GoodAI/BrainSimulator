using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Detect similarity in set")]
    public class LTDetectSimilarity : AbstractLearningTask<RoguelikeWorld>
    {
        protected readonly Random m_rndGen = new Random();
        protected bool m_sameObjectetPlaced;

        public LTDetectSimilarity() : this(null) { }

        public LTDetectSimilarity(SchoolWorld w)
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
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();

            int numberOfObjects = (int)TSHints[TSHintAttributes.NUMBER_OBJECTS];

            m_sameObjectetPlaced = m_rndGen.Next(2) == 0;
            bool placeSameObject = m_sameObjectetPlaced;
            if (m_sameObjectetPlaced)
            {
                numberOfObjects--;
            }

            int numberOfShapes = Enum.GetValues(typeof(Shape.Shapes)).Length;
            List<int> uniqueNumbers = LearningTaskHelpers.UniqueNumbers(m_rndGen, 0, numberOfShapes, numberOfObjects);
            List<Shape.Shapes> shapes = uniqueNumbers.Select(x => (Shape.Shapes)x).ToList();

            foreach (Shape.Shapes shape in shapes)
            {
                SizeF s;
                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1f)
                {
                    float a = (float)(10 + m_rndGen.NextDouble() * 10);
                    s = new SizeF(a, a);
                }
                else
                {
                    s = new Size(15, 15);
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

                PointF position;

                if (placeSameObject)
                {
                    placeSameObject = false;
                    position = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, s, 2);
                    WrappedWorld.CreateShape(shape, color, position, s);
                }

                position = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, s, 2);
                WrappedWorld.CreateShape(shape, color, position, s);
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (m_sameObjectetPlaced == WrappedWorld.Controls.Host[0] > 0)
            {
                wasUnitSuccessful = true;
            }
            return true;
        }
    }
}
