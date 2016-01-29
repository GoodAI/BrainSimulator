using GoodAI.Modules.School.Common;
using System;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectShape : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected Shape.Shapes m_target_type;

        public LTDetectShape() { }

        public LTDetectShape(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.RANDOMNESS_LEVEL, 0},
                {TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 0.5f);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 4);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.0f);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 8);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            World.CreateNonVisibleAgent();

            if (TSHints[TSHintAttributes.IMAGE_NOISE] > 0)
            {
                World.IsImageNoise = true;
            }

            // wtih Pr=.5 show object
            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                //random size
                Size shapeSize = new Size(32, 32);
                if (TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= 1.0f)
                {
                    int side = m_rndGen.Next(10, 48);
                    shapeSize = new Size(side, side);
                }

                // random position
                Point shapePosition = World.Agent.GetGeometry().Location + new Size(20, 0);
                if (TSHints[TSHintAttributes.RANDOMNESS_LEVEL] >= 0.5f)
                {
                    shapePosition = World.RandomPositionInsidePow(m_rndGen, shapeSize);
                }

                m_target_type = Shape.GetRandomShape(m_rndGen, (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]);

                m_target = World.CreateShape(shapePosition, m_target_type, Color.White, shapeSize);
            }
            else
            {
                m_target = null;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // check if two or more controls are active
            int numberOfActiveInputs = (int)World.Controls.Host.Aggregate((x, y) => (float)(Math.Ceiling(x) + Math.Ceiling(y)));
            if (numberOfActiveInputs > 1)
            {
                wasUnitSuccessful = false;
            }
            else if (m_target == null)
            {
                wasUnitSuccessful = numberOfActiveInputs == 0;
            }
            else if (World.Controls.Host[(int)m_target_type] != 0)
            {
                wasUnitSuccessful = true;
            }
            else
            {
                wasUnitSuccessful = false;
            }
            Console.WriteLine(wasUnitSuccessful);
            return true;
        }

    }
}
