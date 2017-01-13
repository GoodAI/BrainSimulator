using GoodAI.Modules.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    [DisplayName("Single pixel - Difficulty 2.1 RL")]
    public class LTSinglePixelTransducerGrayPrecedingWhite : LTSinglePixelTransducerRL
    {

        public LTSinglePixelTransducerGrayPrecedingWhite() : this(null) { }

        public LTSinglePixelTransducerGrayPrecedingWhite(SchoolWorld w)
            : base(w)
        {

        }

        public override void CreateTransducer()
        {
            // this automaton provides rewards if the agent correctly identifies sequences of white symbols that are not preceded by a gray symbol
            m_ft = new FiniteTransducer(3, 3, 2);

            m_ft.SetInitialState(0);

            m_ft.AddFinalState(0);

            m_ft.AddTransition(0, 0, 0, 1);
            m_ft.AddTransition(0, 1, 1, 0);
            m_ft.AddTransition(0, 2, 2, 0);
            m_ft.AddTransition(1, 0, 0, 1);
            m_ft.AddTransition(1, 1, 1, 0);
            m_ft.AddTransition(2, 2, 0, 0);
            m_ft.AddTransition(2, 1, 1, 0);
            m_ft.AddTransition(2, 2, 2, 0);

            m_importantActions.Add(1);
        }
    }
}