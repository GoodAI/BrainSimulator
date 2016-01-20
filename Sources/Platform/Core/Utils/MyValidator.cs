using GoodAI.Core.Execution;
using System.Collections.Generic;

namespace GoodAI.Core.Utils
{
    public interface IValidatable
    {
        string Name { get; }

        void Validate(MyValidator validator);
    }

    // TODO(HonzaS): use this as a dependency instead of MyValidator.
    public interface IValidator
    {
        List<MyValidationMessage> Messages { get; }
        bool ValidationSuccessful { get; }
        MySimulation Simulation { get; set; }
        void ClearValidation();
        void AssertError(bool result, IValidatable sender, string failMessage);
        void AddError(IValidatable sender, string message);
        void AssertWarning(bool result, IValidatable sender, string failMessage);
        void AddWarning(IValidatable sender, string message);
        void AssertInfo(bool result, IValidatable sender, string failMessage);
        void AddInfo(IValidatable sender, string message);
    }

    public class MyValidator
    {
        public List<MyValidationMessage> Messages { get; private set; }    
        public bool ValidationSuccessful { get; private set; }

        public MySimulation Simulation { get; set; }

        public MyValidator()
        {
            Messages = new List<MyValidationMessage>();
            ValidationSuccessful = true;
        }

        public void ClearValidation()
        {
            Messages.Clear();
            ValidationSuccessful = true;
        }

        private void AddMessage(MyValidationLevel level, string message, IValidatable sender) {

            string[] lines = message.Split('\n');

            Messages.Add(new MyValidationMessage(level, lines[0], sender));

            for (int i = 1; i < lines.Length; i++)
            {
                Messages.Add(new MyValidationMessage(level, lines[i], null));
            }
        }        

        public void AssertError(bool result, IValidatable sender, string failMessage) 
        {
            if (!result)
            {
                ValidationSuccessful = false;
                AddMessage(MyValidationLevel.ERROR, failMessage, sender);
            }
        }

        public void AddError(IValidatable sender, string message)
        {
            AssertError(false, sender, message);
        }

        public void AssertWarning(bool result, IValidatable sender, string failMessage) 
        {
            if (!result)
            {
                AddMessage(MyValidationLevel.WARNING, failMessage, sender);
            }
        }

        public void AddWarning(IValidatable sender, string message)
        {
            AssertWarning(false, sender, message);
        }

        public void AssertInfo(bool result, IValidatable sender, string failMessage) 
        {
            if (!result)
            {
                AddMessage(MyValidationLevel.INFO, failMessage, sender);
            }
        }

        public void AddInfo(IValidatable sender, string message)
        {           
            AssertInfo(false, sender, message);            
        }
    }
}
