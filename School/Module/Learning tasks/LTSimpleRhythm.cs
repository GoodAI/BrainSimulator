using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleRhythm : AbstractLearningTask<ManInWorld>
    {
        private TSHintAttribute RHYTHM_MAX_SIZE = new TSHintAttribute("Max number of steps between two ticks", "", typeof(float), 1, 8);
        private TSHintAttribute DELAY = new TSHintAttribute("Maximum random delay between first step and period start", "", typeof(float), 0, 2);

        protected Random m_rndGen = new Random();

        protected enum TimeActions
        {
            NoAction,
            GiveHint,
            AskAction
        }
        protected TimeActions[] m_timeplan;

        public LTSimpleRhythm() : base(null) { }

        public LTSimpleRhythm(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {RHYTHM_MAX_SIZE,  1},

                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(RHYTHM_MAX_SIZE, 2);
            TSProgression.Add(RHYTHM_MAX_SIZE, 4);
            TSProgression.Add(RHYTHM_MAX_SIZE, 6);
            TSProgression.Add(RHYTHM_MAX_SIZE, 8);

        }

        protected override void PresentNewTrainingUnit()
        {
            int rest = m_rndGen.Next(1, (int)TSHints[RHYTHM_MAX_SIZE] + 1);
            m_timeplan = new TimeActions[(rest + 1) * 4];

            //m_timeplan[0]
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = true;
            return true;
        }
    }
}
