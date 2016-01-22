
using System;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    class LTDetectDifference : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected Shape.Shapes m_target_type;

        public LTDetectDifference(ManInWorld w) : base (w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.NOISE, 0},
                {TSHintAttributes.RANDOMNESS, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS, 0.5f);
            TSProgression.Add(TSHintAttributes.RANDOMNESS, 1.0f);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
   
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            return true;
        }

    }
}
