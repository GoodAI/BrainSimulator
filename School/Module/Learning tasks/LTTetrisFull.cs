using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Tetris full")]
    public class LTTetrisFull : AbstractLearningTask<TetrisAdapterWorld>
    {
        public LTTetrisFull() { }

        public LTTetrisFull(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            SetHints(TSHints);
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
