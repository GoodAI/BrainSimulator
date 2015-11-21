using GoodAI.Core.Execution;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Testing.BrainUnit
{
    /// <summary>
    /// Provides access to a snapshot (scan) of a running simulation.
    /// </summary>
    public interface IBrainScan
    {
        float[] GetValues(int nodeId, string blockName = "Output");
    }

    public class BrainScan : IBrainScan
    {
        private readonly MyProjectRunner m_projectRunner;

        public BrainScan(MyProjectRunner runner)
        {
            m_projectRunner = runner;
        }

        public float[] GetValues(int nodeId, string blockName = "Output")
        {
            return m_projectRunner.GetValues(nodeId, blockName);
        }
    }
}
