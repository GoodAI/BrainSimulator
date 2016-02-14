using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayNameAttribute("Simple rhythm")]
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
        private Point m_p;
        private Size m_size;

        private int m_currentStep;

        public LTSimpleRhythm() : base(null) { }

        public LTSimpleRhythm(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {RHYTHM_MAX_SIZE,  1},
                {DELAY, 0},
                {TSHintAttributes.IS_VARIABLE_POSITION, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(DELAY, 1);
            TSProgression.Add(RHYTHM_MAX_SIZE, 2);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(DELAY, 2);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1);
            TSProgression.Add(RHYTHM_MAX_SIZE, 4);
            TSProgression.Add(RHYTHM_MAX_SIZE, 6);
            TSProgression.Add(RHYTHM_MAX_SIZE, 8);
        }

        public override void PresentNewTrainingUnit()
        {
            m_p = WrappedWorld.CreateNonVisibleAgent().GetGeometry().Location;

            m_currentStep = 0;

            int rest = m_rndGen.Next(1, (int)TSHints[RHYTHM_MAX_SIZE] + 1);

            int delay = m_rndGen.Next(0, (int)TSHints[DELAY] + 1);

            m_timeplan = new TimeActions[(rest + 1) * 3 + delay + 1];

            m_timeplan[delay] = TimeActions.GiveHint;
            m_timeplan[delay + (rest + 1)] = TimeActions.GiveHint;
            m_timeplan[delay + (rest + 1) * 2] = TimeActions.AskAction;
            m_timeplan[delay + (rest + 1) * 3] = TimeActions.AskAction;

            m_size = new Size(20, 20);

            if (m_timeplan[0] == TimeActions.GiveHint)
            {
                Point p;
                if (TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1)
                {
                    p = WrappedWorld.RandomPositionInsidePow(m_rndGen, m_size);
                }
                else
                {
                    p = m_p;
                }
                WrappedWorld.CreateShape(p, Shape.Shapes.Tent, Color.White, m_size);
            }
        }

        public override void ExecuteStep()
        {
            if (m_currentStep >= m_timeplan.Length)
            {
                return;
            }

            m_currentStep++;
            base.ExecuteStep();
            WrappedWorld.gameObjects.Clear();
            if (m_timeplan[m_currentStep] == TimeActions.GiveHint)
            {
                Point p;

                if (TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1)
                {
                    p = WrappedWorld.RandomPositionInsidePow(m_rndGen, m_size);
                }
                else
                {
                    p = m_p;
                }

                WrappedWorld.CreateShape(p, Shape.Shapes.Tent, Color.White, m_size);
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;

            if (!((WrappedWorld.Controls.Host[0] != 0 && m_timeplan[m_currentStep] == TimeActions.AskAction)
            || (WrappedWorld.Controls.Host[0] == 0 && m_timeplan[m_currentStep] != TimeActions.AskAction)))
            {
                return true;
            }

            if (m_currentStep + 1 == m_timeplan.Length)
            {
                wasUnitSuccessful = true;
                return true;
            }
            return false;
        }
    }
}
