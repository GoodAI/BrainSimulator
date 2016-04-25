using System;

namespace Logger
{
    public class TWLogMessage
    {
        public TWSeverity Severity { get; private set; }
        public string Template { get; private set; }
        public object[] Objects { get; private set; }
        public Exception Exception { get; private set; }

        public TWLogMessage(TWSeverity severity, string template, object[] objects, Exception exception)
        {
            Severity = severity;
            Template = template;
            Objects = objects;
            Exception = exception;
        }

        public override string ToString()
        {
            return Template + " " + Exception;
        }
    }
}