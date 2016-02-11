using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTPongTest : AbstractLearningTask<PongAdapterWorld>
    {
        public LTPongTest() : base(null) { }

        public LTPongTest(SchoolWorld w)
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
