using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;

namespace GoodAI.Platform.Core.Nodes
{
    public interface IModelChanger
    {
        bool IsModelChanging { get; }
        void ChangeModel(ref List<MyWorkingNode> removedNodes, ref List<MyWorkingNode> addedNodes);
    }
}
