using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTCompatibilityMatching : AbstractLearningTask<TetrisAdapterWorld>
    {
        public LTCompatibilityMatching() : base(null) { }

        public static readonly TSHintAttribute ROTATION_ALLOWED = new TSHintAttribute("ROTATION_ALLOWED", "", typeof(bool), 0, 1);

        public LTCompatibilityMatching(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { ROTATION_ALLOWED, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(ROTATION_ALLOWED, 1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.Engine.ResetToRandomHorizon();
            for (int i = 0; i < 4 * WrappedWorld.WaitStepsPerFall; i++)                
                WrappedWorld.Engine.Step(Modules.TetrisWorld.TetrisWorld.ActionInputType.NoAction);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (TSHints[ROTATION_ALLOWED] > 0)
            {
                    wasUnitSuccessful = wasUnitSuccessful || WrappedWorld.Engine.CanMatch();
                    wasUnitSuccessful = wasUnitSuccessful || WrappedWorld.Engine.CanMatch(TetrisWorld.TetrominoRotation.Left);
                    wasUnitSuccessful = wasUnitSuccessful || WrappedWorld.Engine.CanMatch(TetrisWorld.TetrominoRotation.Right);
                    wasUnitSuccessful = wasUnitSuccessful || WrappedWorld.Engine.CanMatch(TetrisWorld.TetrominoRotation.UpsideDown);
            }
            else
            {
                wasUnitSuccessful = WrappedWorld.Engine.CanMatch();
            }
            return true;
        }
    }
}
