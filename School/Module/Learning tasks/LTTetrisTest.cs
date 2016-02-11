using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTTetrisTest : AbstractLearningTask<TetrisAdapterWorld>
    {
        public LTTetrisTest() : base(null) { }

        public LTTetrisTest(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
        }

        public override void PresentNewTrainingUnit()
        {

        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            return false;
        }
    }
}
