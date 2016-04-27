using System;

namespace Logger
{
    public class LogMessage
    {
        public Severity Severity { get; private set; }
        public string Template { get; private set; }
        public object[] Objects { get; private set; }
        public Exception Exception { get; private set; }
        public DateTime Time;

        public LogMessage(Severity severity, string template, object[] objects, Exception exception)
        {
            Severity = severity;
            Template = template;
            Objects = objects;
            Exception = exception;
            Time = DateTime.Now;
        }
    }
}
