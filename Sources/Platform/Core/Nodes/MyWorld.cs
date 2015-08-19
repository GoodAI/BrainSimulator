using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public abstract class MyWorld : MyWorkingNode
    {
        public virtual void DoPause()
        {

        }

        public void ValidateWorld(MyValidator validator)
        {
            ValidateMandatory(validator);
            Validate(validator);
        }

        public override void ProcessOutgoingSignals()
        {
            OutgoingSignals = 0;

            OutgoingSignals |= RiseSignalMask;
            OutgoingSignals &= ~DropSignalMask;
        }
    }
}