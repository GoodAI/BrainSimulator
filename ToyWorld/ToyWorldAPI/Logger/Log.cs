using System;
using System.Collections.Concurrent;
using GoodAI.Logging;

namespace GoodAI.ToyWorldAPI.Logger
{
    /// <summary>
    /// Logger. Call Log.Instance.Method(). Implemented as queue (FIFO).
    /// </summary>
    public class Log : ILog, ILoggerDequeuer
    {
        private static Log m_log;

        private readonly ConcurrentQueue<LogMessage> m_queue = new ConcurrentQueue<LogMessage>();

        private readonly string[] m_severityNames = typeof(Severity).GetEnumNames();

        public void Add(Severity severity, string template, params object[] objects)
        {
            m_queue.Enqueue(new LogMessage(severity, template, objects, null));
        }

        public void Add(Severity severity, Exception ex, string template, params object[] objects)
        {
            m_queue.Enqueue(new LogMessage(severity, template, objects, ex));
        }

        public Tuple<string, object[], Exception> Deqeue()
        {
            LogMessage message; 
            m_queue.TryDequeue(out message);
            if (message == null)
            {
                return null;
            }
            return new Tuple<string, object[], Exception>(
                m_severityNames[(int)message.Severity] + message.Template,
                message.Objects,
                message.Exception
                );
        }

        /// <summary>
        /// Singleton
        /// </summary>
        public static Log Instance
        {
            get
            {
                if (m_log == null)
                {
                    m_log = new Log();
                }
                return m_log;
            }
        }
    }
}