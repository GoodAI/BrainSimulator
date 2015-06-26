using BrainSimulator.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.Utils
{
    public enum MyValidationLevel
    {
        INFO = 0,
        WARNING = 1,
        ERROR = 2
    }

    public class MyValidationMessage
    {
        public MyValidationLevel Level { get; private set; }
        public string Message { get; private set; }
        public MyNode Sender { get; private set; }

        internal MyValidationMessage(MyValidationLevel level, string message, MyNode sender)
        {
            Level = level;
            Message = message;
            Sender = sender;
        }
    }
}
