using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 5 RL")]
    public class LTSinglePixelTransducerDivBy5 : LTSinglePixelTransducerRL
    {

        public LTSinglePixelTransducerDivBy5() : this(null) { }

        public LTSinglePixelTransducerDivBy5(SchoolWorld w)
            : base(w)
        {

        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies binary strings representing numbers divisible by 5
            m_ft = new FiniteTransducer(5, 2, 2);

            m_ft.SetInitialState(0);

            m_ft.AddFinalState(0);

            m_ft.AddTransition(0, 0, 0, 1);
            m_ft.AddTransition(0, 1, 1, 0);
            m_ft.AddTransition(1, 2, 0, 0);
            m_ft.AddTransition(1, 3, 1, 0);
            m_ft.AddTransition(2, 4, 0, 0);
            m_ft.AddTransition(2, 0, 1, 1);
            m_ft.AddTransition(3, 1, 0, 0);
            m_ft.AddTransition(3, 2, 1, 0);
            m_ft.AddTransition(4, 3, 0, 0);
            m_ft.AddTransition(4, 4, 1, 0);

            m_importantActions.Add(1);
        }
    }
}