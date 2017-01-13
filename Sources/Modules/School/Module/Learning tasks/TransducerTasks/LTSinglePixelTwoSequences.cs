using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 1.1 RL")]
    public class LTSinglePixelTwoSequences : LTSinglePixelTransducerRL
    {

        public LTSinglePixelTwoSequences() : this(null) { }

        public LTSinglePixelTwoSequences(SchoolWorld w)
            : base(w)
        {

        }

        public override int NumberOfSuccessesRequired
        {
            get { return 10; }
        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies binary strings representing numbers divisible by two
            m_ft = new FiniteTransducer(7, 3, 2);

            m_ft.SetInitialState(0);

            m_ft.AddFinalState(2);
            m_ft.AddFinalState(6);

            m_ft.AddTransition(0, 1, 0, 0);
            m_ft.AddTransition(1, 2, 0, 1);
            m_ft.AddTransition(2, 3, 1, 0);
            m_ft.AddTransition(3, 0, 2, 0);
            m_ft.AddTransition(0, 4, 1, 0);
            m_ft.AddTransition(4, 5, 0, 0);
            m_ft.AddTransition(5, 6, 1, 1);
            m_ft.AddTransition(6, 0, 2, 0);

            m_importantActions.Add(1);
        }
    }
}