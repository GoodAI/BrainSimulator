
namespace GoodAI.Modules.Harm
{
    /// <summary>
    /// Produces motivation to execute the current strategy (abstract) - action.
    /// </summary>
    public class MyMotivationSource
    {
        private float m_val;
        private MyModuleParams m_setup;
        private int m_level;

        public MyMotivationSource(MyModuleParams setup, int level)
        {
            this.m_setup = setup;
            this.m_val = 0;
            this.m_level = level;
        }

        /// <summary>
        /// Different dynamics on different levels
        /// </summary>
        public void MakeStep()
        {
            m_val += m_setup.MotivationChange + (m_level * m_setup.HierarchicalMotivationScale);
            if (m_val > 1)
            {
                m_val = 1;
            }
            else if (m_val < -1)
            {
                m_val = -1;
            }
        }

        public float GetMotivation()
        {
            return m_val;
        }

        public void OverrideWith(float val)
        {
            if (val > 1)
            {
                val = 1;
            }
            else if (val < 0)
            {
                val = 0;
            }
            this.m_val = val;
        }

        public void Reset()
        {
            this.m_val = 0;
        }
    }
}
