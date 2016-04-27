using System.Collections.Concurrent;

namespace Logger
{
    public class AbstractLog
    {
        protected readonly ConcurrentQueue<LogMessage> Queue = new ConcurrentQueue<LogMessage>();

        protected readonly string[] SeverityNames = typeof(Severity).GetEnumNames();
    }
}
