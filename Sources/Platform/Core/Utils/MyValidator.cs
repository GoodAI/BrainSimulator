using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Utils
{
    public interface IValidable
    {
        string Name { get; set; }

        void Validate(MyValidator validator);
    }

    public class MyValidator
    {
        public List<MyValidationMessage> Messages { get; private set; }    
        public bool ValidationSucessfull { get; private set; }

        public MySimulation Simulation { get; set; }

        public MyValidator()
        {
            Messages = new List<MyValidationMessage>();
            ValidationSucessfull = true;
        }

        public void ClearValidation()
        {
            Messages.Clear();
            ValidationSucessfull = true;
        }

        private void AddMessage(MyValidationLevel level, string message, IValidable sender) {

            string[] lines = message.Split('\n');

            Messages.Add(new MyValidationMessage(level, lines[0], sender));

            for (int i = 1; i < lines.Length; i++)
            {
                Messages.Add(new MyValidationMessage(level, lines[i], null));
            }
        }        

        public void AssertError(bool result, IValidable sender, string failMessage) 
        {
            if (!result)
            {
                ValidationSucessfull = false;
                AddMessage(MyValidationLevel.ERROR, failMessage, sender);
            }
        }

        public void AddError(IValidable sender, string message)
        {
            AssertError(false, sender, message);
        }

        public void AssertWarning(bool result, IValidable sender, string failMessage) 
        {
            if (!result)
            {
                AddMessage(MyValidationLevel.WARNING, failMessage, sender);
            }
        }

        public void AddWarning(IValidable sender, string message)
        {
            AssertWarning(false, sender, message);
        }

        public void AssertInfo(bool result, IValidable sender, string failMessage) 
        {
            if (!result)
            {
                AddMessage(MyValidationLevel.INFO, failMessage, sender);
            }
        }

        public void AddInfo(IValidable sender, string message)
        {           
            AssertInfo(false, sender, message);            
        }
    }
}
