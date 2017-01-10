using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 1 RL")]
    public class LTSinglePixelFABackedDivBy2 : LTSinglePixelTransducerRL
    {

        public LTSinglePixelFABackedDivBy2() : this(null) { }

        public LTSinglePixelFABackedDivBy2(SchoolWorld w)
            : base(w)
        {

        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies binary strings representing numbers divisible by two
            m_ft = new FiniteTransducer(2, 2, 2);

            m_ft.SetInitialState(0);

            m_ft.AddFinalState(0);

            m_ft.AddTransition(0, 0, 0, 1);
            m_ft.AddTransition(0, 1, 1, 0);
            m_ft.AddTransition(1, 0, 0, 1);
            m_ft.AddTransition(1, 1, 1, 0);

            m_importantActions.Add(1);
        }
    }
}