using GoodAI.Core.Utils;

namespace GoodAI.Core.Nodes
{
    public abstract class MyWorld : MyWorkingNode
    {
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