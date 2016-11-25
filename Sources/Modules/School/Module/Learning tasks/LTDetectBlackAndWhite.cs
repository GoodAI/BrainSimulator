using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    // TODO:
    // World actions have not been implemented yet.
    // Multiple parameters are incremented in the same step.

    [DisplayName("Detect objects presence")]
    public class LTDetectBlackAndWhite : AbstractLearningTask<RoguelikeWorld>
    {
        private static readonly TSHintAttribute IS_TARGET_MOVING = new TSHintAttribute("Is target moving", "", typeof(bool), 0, 1);

        private readonly Random m_rndGen = new Random();
        private bool m_appears;
        private bool m_isBlack;

        public LTDetectBlackAndWhite() : this(null) { }

        public LTDetectBlackAndWhite(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { IS_TARGET_MOVING, 0 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(
                new TrainingSetHints {
                    { TSHintAttributes.IS_VARIABLE_SIZE, 1 },
                    { TSHintAttributes.IMAGE_NOISE, 1 },
                    { IS_TARGET_MOVING, 1 }
                });
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();

            m_appears = LearningTaskHelpers.FlipCoin(m_rndGen);
            if (!m_appears) return;

            SizeF size;
            if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1)
            {
                float a = (float)(10 + m_rndGen.NextDouble() * 10);
                size = new SizeF(a, a);
            }
            else
            {
                size = new Size(15, 15);
            }

            PointF position;
            if (TSHints[IS_TARGET_MOVING] >= 1)
            {
                position = WrappedWorld.RandomPositionInsideViewport(m_rndGen, size);
            }
            else
            {
                position = WrappedWorld.Agent.GetGeometry().Location;
            }

            m_isBlack = LearningTaskHelpers.FlipCoin(m_rndGen);
            Color color = m_isBlack ? Color.Black : Color.White;

            WrappedWorld.CreateShape(Shape.Shapes.Square, color, position, size);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if ((!m_appears && WrappedWorld.Controls.Host[0] == 0 && WrappedWorld.Controls.Host[1] == 0) ||
                (m_isBlack && WrappedWorld.Controls.Host[0] != 0 && WrappedWorld.Controls.Host[1] == 0) ||
                (!m_isBlack && WrappedWorld.Controls.Host[0] == 0 && WrappedWorld.Controls.Host[1] != 0)
                )
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

    // The learning task consists of training units (TUs).
    // There is a fixed number of difficulty levels (currently 2).
    // We call the TUs associated with a level a training set.
    //
    // If the agent successfully concludes a training set,
    // the learning task proceeds to the next difficulty level.
    // If the agent fails to conclude the training set
    // within a limited number of attempts, the learning task
    // exits with failure.
    //
    // In pseudocode:
    //
    // start at lowest level of difficulty
    // for each level of difficulty (training set)
    //   for a limited number of examples (training units)
    //      present example
    //      reward / punish
    //      if number of successful tests at level == requirement
    //          if level of difficulty == highest level
    //              exit learning task with success (ability learned)
    //          else
    //              proceed to next level
    //   exit learning task with failure
    //
    //
}
