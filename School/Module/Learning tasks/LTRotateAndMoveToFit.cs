using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Rotate and move to fit")]
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
            WrappedWorld.Engine.Step(Modules.TetrisWorld.TetrisWorld.ActionInputType.NoAction);
            while (!WrappedWorld.Engine.CanMatchAnyRotation())
            {
                WrappedWorld.Engine.ResetToRandomHorizon();
                WrappedWorld.Engine.Step(Modules.TetrisWorld.TetrisWorld.ActionInputType.NoAction);
            }
            for (int i = 0; i < 4 * WrappedWorld.WaitStepsPerFall; i++)
                WrappedWorld.Engine.Step(Modules.TetrisWorld.TetrisWorld.ActionInputType.NoAction);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = WrappedWorld.Engine.IsMerging && WrappedWorld.Engine.CleanMatch;
            return WrappedWorld.Engine.IsMerging;
        }
    }
}
