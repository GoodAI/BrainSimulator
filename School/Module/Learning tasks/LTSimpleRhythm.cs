using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleRhythm : AbstractLearningTask<ManInWorld>
    {
        private TSHintAttribute RHYTHM_MAX_SIZE = new TSHintAttribute("Max number of steps between two ticks", "", typeof(int), 1, 8);
        private TSHintAttribute DELAY = new TSHintAttribute("Maximum random delay between first step and period start", "", typeof(int), 0, 2);

        protected Random m_rndGen = new Random();

        protected enum TimeActions
        {
            NoAction = 0,
            GiveHint,
            AskAction
        }
        protected TimeActions[] m_timeplan;
        private Point p;

        int m_currentStep;

        public LTSimpleRhythm() : base(null) { }

        public LTSimpleRhythm(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {RHYTHM_MAX_SIZE,  1},
                {DELAY, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(DELAY, 1);
            TSProgression.Add(RHYTHM_MAX_SIZE, 2);
            TSProgression.Add(DELAY, 2);
            TSProgression.Add(RHYTHM_MAX_SIZE, 4);
            TSProgression.Add(RHYTHM_MAX_SIZE, 6);
            TSProgression.Add(RHYTHM_MAX_SIZE, 8);

        }

        public override void PresentNewTrainingUnit()
        {
            p = WrappedWorld.CreateNonVisibleAgent().GetGeometry().Location;

            m_currentStep = 0;

            int rest = m_rndGen.Next(1, (int)TSHints[RHYTHM_MAX_SIZE] + 1);

            int delay = m_rndGen.Next(0, (int)TSHints[DELAY] + 1);

            m_timeplan = new TimeActions[(rest + 1) * 3 + delay + 1];

            m_timeplan[delay] = TimeActions.GiveHint;
            m_timeplan[delay + (rest + 1)] = TimeActions.GiveHint;
            m_timeplan[delay + (rest + 1) * 2] = TimeActions.AskAction;
            m_timeplan[delay + (rest + 1) * 3] = TimeActions.AskAction;

            if (m_timeplan[0] == TimeActions.GiveHint)
            {
                WrappedWorld.CreateShape(p, Shape.Shapes.Tent);
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            WrappedWorld.ClearWorld();
            if (m_timeplan[m_currentStep] == TimeActions.GiveHint)
            {
                WrappedWorld.CreateShape(p, Shape.Shapes.Tent);
            }

            wasUnitSuccessful = false;

            if (!((WrappedWorld.Controls.Host[0] != 0 && m_timeplan[m_currentStep] == TimeActions.AskAction)
            || (WrappedWorld.Controls.Host[0] == 0 && m_timeplan[m_currentStep] != TimeActions.AskAction)))
            {
                wasUnitSuccessful = false;
                return true;
            }

            if (m_currentStep + 1 == m_timeplan.Length)
            {
                return true;
            }

            
            return false;
        }
    }
}
