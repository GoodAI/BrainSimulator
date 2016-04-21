using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Logging;

namespace GoodAI.ToyWorldAPI.Logger
{
    class LogMessage
    {
        public readonly Severity Severity;
        public readonly string Template;
        public readonly object[] Objects;
        public readonly Exception Exception;

        public LogMessage(Severity severity, string template, object[] objects, Exception exception)
        {
            Severity = severity;
            Template = template;
            Objects = objects;
            Exception = exception;
        }
    }
}
