using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTShapeGroups : AbstractLearningTask<RoguelikeWorld>
    {
        public LTShapeGroups() { }

        public LTShapeGroups(SchoolWorld w)
            : base(w)
        {
            //  TODO Add TSHints

            TSProgression.Add(TSHints.Clone());
            // TODO progression
            SetHints(TSHints);
        }

        protected override void SetHints(TrainingSetHints trainingSetHints)
        {
            WrappedWorld.SetHints(trainingSetHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            // TODO
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // TODO
            wasUnitSuccessful = true;
            return true;
        }
    }
}
