using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 3 RL")]
    public class LTSinglePixelTransducerDivBy3 : LTSinglePixelTransducerRL
    {

        public LTSinglePixelTransducerDivBy3() : this(null) { }

        public LTSinglePixelTransducerDivBy3(SchoolWorld w)
            : base(w)
        {

        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies binary strings representing numbers divisible by 3
            m_ft = new FiniteTransducer(3, 2, 2);

            m_ft.SetInitialState(0);

            m_ft.AddFinalState(0);

            m_ft.AddTransition(0, 0, 0, 1);
            m_ft.AddTransition(0, 1, 1, 0);
            m_ft.AddTransition(1, 0, 1, 1);
            m_ft.AddTransition(1, 2, 0, 0);
            m_ft.AddTransition(2, 1, 0, 0);
            m_ft.AddTransition(2, 2, 1, 0);

            m_importantActions.Add(1);
        }
    }
}