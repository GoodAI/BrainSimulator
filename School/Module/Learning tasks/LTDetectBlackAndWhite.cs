using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    // TODO:
    // World actions have not been implemented yet.
    // Multiple parameters are incremented in the same step.

    public class LTDetectBlackAndWhite : AbstractLearningTask<ManInWorld>
    {
        private static readonly TSHintAttribute IS_TARGET_MOVING = new TSHintAttribute("Is target moving", "", typeof(bool), 0, 1);

        public LTDetectBlackAndWhite() { }
        private Random m_rndGen = new Random();
        private bool m_appears;
        private bool m_isBlack;

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

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            if (WrappedWorld.GetType() != typeof(RoguelikeWorld))
            {
                throw new NotImplementedException();
            }

            WrappedWorld.CreateNonVisibleAgent();

            m_appears = LearningTaskHelpers.FlipCoin(m_rndGen);
            if (!m_appears) return;

            Size size;
            if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1)
            {
                int a = m_rndGen.Next(10,20);
                size = new Size(a, a);
            }
            else
            {
                size = new Size(15, 15);
            }

                        Point position;
            if (TSHints[IS_TARGET_MOVING] >= 1)
            {
                position = WrappedWorld.RandomPositionInsidePow(m_rndGen, size);
            }
            else
            {
                position = WrappedWorld.Agent.GetGeometry().Location;
            }

            m_isBlack = LearningTaskHelpers.FlipCoin(m_rndGen);
            Color color = m_isBlack ? Color.Black : Color.White;

            WrappedWorld.CreateShape(position, Shape.Shapes.Square, color, size);
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
