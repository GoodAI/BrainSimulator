using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectShapeColor : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected Shape.Shapes m_target_type;

        public LTDetectShapeColor() : base(null) { }

        public LTDetectShapeColor(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { TSHintAttributes.IS_VARIABLE_POSITION, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
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
                    shapePosition = WrappedWorld.RandomPositionInsidePow(m_rndGen, shapeSize);
                }

                // random color
                Color shapeColor = LearningTaskHelpers.FlipCoin(m_rndGen) ? Color.Cyan : Color.Yellow;

                m_target_type = LearningTaskHelpers.FlipCoin(m_rndGen) ? Shape.Shapes.Circle : Shape.Shapes.Square;

                m_target = WrappedWorld.CreateShape(shapePosition, m_target_type, shapeColor, shapeSize);
            }
            else
            {
                m_target = null;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            bool wasCircleTargetDetected = WrappedWorld.Controls.Host[(int)Shape.Shapes.Circle] != 0;
            bool wasSquareTargetDetected = (WrappedWorld as ManInWorld).Controls.Host[(int)Shape.Shapes.Square] != 0;

            // both target detected
            if (wasCircleTargetDetected && wasSquareTargetDetected)
            {
                wasUnitSuccessful = false;
                GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
                return true;
            }

            // no target
            if (m_target == null)
            {
                if (wasSquareTargetDetected || wasCircleTargetDetected)
                {
                    wasUnitSuccessful = false;
                }
                else
                {
                    wasUnitSuccessful = true;
                }
                GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
                return true;
            }

            wasUnitSuccessful = (wasCircleTargetDetected && m_target_type == Shape.Shapes.Circle ||
                wasSquareTargetDetected && m_target_type == Shape.Shapes.Square);

            GoodAI.Core.Utils.MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
            return true;
        }
    }
}
