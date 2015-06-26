using BrainSimulator.Execution;
using BrainSimulator.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.Utils
{
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

        private void AddMessage(MyValidationLevel level, string message, MyNode sender) {

            string[] lines = message.Split('\n');

            Messages.Add(new MyValidationMessage(level, lines[0], sender));

            for (int i = 1; i < lines.Length; i++)
            {
                Messages.Add(new MyValidationMessage(level, lines[i], null));
            }
        }        

        public void AssertError(bool result, MyNode sender, string failMessage) 
        {
            if (!result)
            {
                ValidationSucessfull = false;
                AddMessage(MyValidationLevel.ERROR, failMessage, sender);
            }
        }

        public void AddError(MyNode sender, string message)
        {
            AssertError(false, sender, message);
        }

        public void AssertWarning(bool result, MyNode sender, string failMessage) 
        {
            if (!result)
            {
                AddMessage(MyValidationLevel.WARNING, failMessage, sender);
            }
        }

        public void AddWarning(MyNode sender, string message)
        {
            AssertWarning(false, sender, message);
        }

        public void AssertInfo(bool result, MyNode sender, string failMessage) 
        {
            if (!result)
            {
                AddMessage(MyValidationLevel.INFO, failMessage, sender);
            }
        }

        public void AddInfo(MyNode sender, string message)
        {           
            AssertInfo(false, sender, message);            
        }
    }
}
