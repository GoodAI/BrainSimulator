using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Nodes
{
    /// <summary>
    /// Implement this if custom names of variable input/output branches in GUI are desired (rather than default numbering).
    /// </summary>
    public interface IMyVariableBranchViewWithNamesNode : IMyVariableBranchViewNodeBase
    {
        /// <summary>
        /// Get name (for GUI) of a given input branch
        /// </summary>
        /// <param name="index"></param>
        /// <returns>name or null</returns>
        String GetInputBranchName(int index);

        /// <summary>
        /// Get name (for GUI) of a given output branch
        /// </summary>
        /// <param name="index"></param>
        /// <returns>name or null</returns>
        String GetOutputBranchName(int index);
    }
}
