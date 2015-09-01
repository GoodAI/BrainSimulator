
namespace GoodAI.Core.Utils
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
        public IValidatable Sender { get; private set; }

        internal MyValidationMessage(MyValidationLevel level, string message, IValidatable sender)
        {
            Level = level;
            Message = message;
            Sender = sender;
        }
    }
}
