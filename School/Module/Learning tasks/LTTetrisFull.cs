using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.School.Worlds;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
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

        protected override void PresentNewTrainingUnit()
        {

        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            return false;
        }
    }
}
