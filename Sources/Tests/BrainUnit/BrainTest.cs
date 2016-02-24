using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Testing.BrainUnit
{
    public abstract class BrainTest
    {
        public string BrainFileName { get; protected set; }
        public int MaxStepCount { get; protected set; }
        public bool ExpectedToFail { get; protected set; }

        public virtual int InspectInterval
        {
            get  // TODO(Premek): unit test
            {
                if (m_inspectInterval > 0)
                    return m_inspectInterval;

                return Math.Max(1, Math.Min(100, MaxStepCount / 10));
            }
            protected set
            {
                m_inspectInterval = value;
            }
        }
        private int m_inspectInterval = -1;

        public virtual string Name
        {
            get { return GetType().Name; }
        }

        public virtual bool ShouldStop(IBrainScan b)
        {
            return false;
        }

        public abstract void Check(IBrainScan b);
    }
}
