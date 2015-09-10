using GoodAI.Core.Utils;

namespace GoodAI.Core.Nodes
{
    public abstract class MyWorld : MyWorkingNode
    {
        public virtual void Cleanup() 
        {
        
        }

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