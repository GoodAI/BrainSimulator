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
    public class LTRotateAndMoveToFit : AbstractLearningTask<TetrisAdapterWorld>
    {
        public LTRotateAndMoveToFit() { }

        public LTRotateAndMoveToFit(SchoolWorld w)
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
            WrappedWorld.Engine.ResetToRandomHorizon();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = WrappedWorld.Engine.IsMerging && WrappedWorld.Engine.CleanMatch;
            return WrappedWorld.Engine.IsMerging;
        }
    }
}
