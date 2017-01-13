using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 4 RL")] // Rem03 means "with remainder 0 or 3"
    public class LTSinglePixelTransducerDivBy5Rem03 : LTSinglePixelTransducerRL
    {

        public LTSinglePixelTransducerDivBy5Rem03() : this(null) { }

        public LTSinglePixelTransducerDivBy5Rem03(SchoolWorld w)
            : base(w)
        {

        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies binary strings representing numbers divisible by 5 (or with remainder 3)
            m_ft = new FiniteTransducer(5, 2, 2);

            m_ft.SetInitialState(0);

            m_ft.AddFinalState(0);
            m_ft.AddFinalState(3); // this means this is no longer division by 5, but also division by 5 with remainder 3

            m_ft.AddTransition(0, 0, 0, 1);
            m_ft.AddTransition(0, 1, 1, 0);
            m_ft.AddTransition(1, 2, 0, 0);
            m_ft.AddTransition(1, 3, 1, 1); // 3 is now a final state
            m_ft.AddTransition(2, 4, 0, 0);
            m_ft.AddTransition(2, 0, 1, 1);
            m_ft.AddTransition(3, 1, 0, 0);
            m_ft.AddTransition(3, 2, 1, 0);
            m_ft.AddTransition(4, 3, 0, 1); // 3 is now a final state
            m_ft.AddTransition(4, 4, 1, 0);

            m_importantActions.Add(1);
        }
    }
}