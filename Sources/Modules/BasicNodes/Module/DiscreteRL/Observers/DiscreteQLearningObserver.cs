using GoodAI.Core.Nodes;
using GoodAI.Modules.DiscreteRL.Observers;
using GoodAI.Modules.Harm;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.DiscreteRL.Observers
{
    /// <summary>
    /// Observer for the DiscreteQLearning, specifies the generic type of DiscretePolicyObserver.
    /// </summary>
    class DiscreteQLearningObserver : DiscretePolicyObserver<MyDiscreteQLearningNode>
    {
    }
}
