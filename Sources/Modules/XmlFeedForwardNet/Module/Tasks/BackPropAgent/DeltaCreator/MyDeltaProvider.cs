using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using  XmlFeedForwardNet.Networks;

namespace  XmlFeedForwardNet.Tasks.BackPropAgent.DeltaCreator
{
    public abstract class MyDeltaProvider
    {
        protected MyAbstractFeedForwardNode m_network;

        public MyDeltaProvider(MyAbstractFeedForwardNode network)
        {
            m_network = network;
        }
        
        public abstract void Execute();
    }
}
