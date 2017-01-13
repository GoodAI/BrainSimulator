using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 2 RL")]
    public class LTSinglePixelTransducerDivBy4 : LTSinglePixelTransducerRL
    {

        public LTSinglePixelTransducerDivBy4() : this(null) { }

        public LTSinglePixelTransducerDivBy4(SchoolWorld w)
            : base(w)
        {

        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies binary strings representing numbers divisible by 4
            m_ft = new FiniteTransducer(3, 2, 2);

            m_ft.SetInitialState(1);

            m_ft.AddFinalState(0);

            m_ft.AddTransition(0, 0, 0, 1);
            m_ft.AddTransition(0, 1, 1, 0);
            m_ft.AddTransition(1, 1, 1, 0);
            m_ft.AddTransition(1, 2, 0, 0);
            m_ft.AddTransition(2, 1, 1, 0);
            m_ft.AddTransition(2, 0, 0, 1);

            m_importantActions.Add(1);
        }
    }
}