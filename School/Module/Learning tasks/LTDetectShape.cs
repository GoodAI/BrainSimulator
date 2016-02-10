using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
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

        public LTDetectShape() : base(null) { }

        public LTDetectShape(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.IS_VARIABLE_COLOR, 0},
                {TSHintAttributes.IS_VARIABLE_SIZE, 0},
                {TSHintAttributes.IS_VARIABLE_POSITION, 0},
                {TSHintAttributes.IS_VARIABLE_ROTATION, 0},
                {TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1.0f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1.0f);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 4);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1.0f);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 8);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_ROTATION, 1.0f);
        }

        protected override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();

            // wtih Pr=.5 show object
            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                //random size
                Size shapeSize = new Size(32, 32);
                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1.0f)
                {
                    int side = m_rndGen.Next(10, 48);
                    shapeSize = new Size(side, side);
                }

                // random position
                Point shapePosition = WrappedWorld.Agent.GetGeometry().Location + new Size(20, 0);
                if (TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1.0f)
                {
                    shapePosition = WrappedWorld.RandomPositionInsidePow(m_rndGen, shapeSize, 2);
                }

                // random color
                Color shapeColor = Color.White;
                if (TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1.0f)
                {
                    shapeColor = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
                }

                // random rotation
                float rotation = 0;
                if (TSHints[TSHintAttributes.IS_VARIABLE_ROTATION] >= 1.0f)
                {
                    rotation = (float)(m_rndGen.NextDouble() * 2 * Math.PI);
                }

                m_target_type = Shape.GetRandomShape(m_rndGen, (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]);

                m_target = WrappedWorld.CreateShape(shapePosition, m_target_type, shapeColor, shapeSize, rotation: rotation);
            }
            else
            {
                m_target = null;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // check if two or more controls are active
            int numberOfActiveInputs = (int)WrappedWorld.Controls.Host.Aggregate((x, y) => (float)(Math.Ceiling(x) + Math.Ceiling(y)));
            if (numberOfActiveInputs > 1)
            {
                wasUnitSuccessful = false;
            }
            else if (m_target == null)
            {
                wasUnitSuccessful = numberOfActiveInputs == 0;
            }
            else if (WrappedWorld.Controls.Host[(int)m_target_type] != 0)
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
