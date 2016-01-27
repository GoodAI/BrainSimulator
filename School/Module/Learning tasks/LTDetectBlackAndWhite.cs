using GoodAI.Modules.School.Common;

namespace GoodAI.Modules.School.LearningTasks
{
    // TODO:
    // World actions have not been implemented yet.
    // Multiple parameters are incremented in the same step.

    public class LTDetectBlackAndWhite : AbstractLearningTask<ManInWorld>
    {
        public LTDetectBlackAndWhite() { }

        public LTDetectBlackAndWhite(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { TSHintAttributes.TARGET_MIN_SIZE, 1 },
                { TSHintAttributes.TARGET_MAX_SIZE, 3 },
                { TSHintAttributes.NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { TSHintAttributes.IS_TARGET_MOVING, 0 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(
                new TrainingSetHints {
                    { TSHintAttributes.TARGET_MAX_SIZE, 3 },
                    { TSHintAttributes.NOISE, 1 },
                    { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100 },
                    { TSHintAttributes.IS_TARGET_MOVING, 1 }
                });

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {

        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            return false;
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
