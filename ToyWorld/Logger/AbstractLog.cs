using System.Collections.Concurrent;

namespace Logger
{
    public class AbstractLog
    {
        protected readonly ConcurrentQueue<LogMessage> m_queue = new ConcurrentQueue<LogMessage>();

        protected readonly string[] m_severityNames = typeof(Severity).GetEnumNames();
    }
}