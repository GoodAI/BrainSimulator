using System.Collections.Concurrent;
using Logger;

namespace GoodAI.Logging
{
    public class AbstractLog
    {
        protected readonly ConcurrentQueue<LogMessage> m_queue = new ConcurrentQueue<LogMessage>();

        protected readonly string[] m_severityNames = typeof(Severity).GetEnumNames();
    }
}