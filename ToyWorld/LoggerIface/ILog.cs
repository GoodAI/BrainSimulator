using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Logging
{
    public enum Severity
    {
        Error,
        Warn,
        Info,
        Debug,
        Verbose
    }

    public interface ILog
    {
        void Add(Severity severity, string template, params object[] objects);
        void Add(Severity severity, Exception ex, string template, params object[] objects);
    }
}
