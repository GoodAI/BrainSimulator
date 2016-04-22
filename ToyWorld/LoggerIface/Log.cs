using System;
using Logger;

namespace GoodAI.Logging
{
    /// <summary>
    /// Logger. Call Log.Instance.Method(). Implemented as queue (FIFO).
    /// </summary>
    public class Log : AbstractLog, ILog
    {
        private static Log m_log;

        public void Add(Severity severity, string template, params object[] objects)
        {
            Queue.Enqueue(new LogMessage(severity, template, objects, null));
        }

        public void Add(Severity severity, Exception ex, string template, params object[] objects)
        {
            Queue.Enqueue(new LogMessage(severity, template, objects, ex));
        }

        /// <summary>
        /// Singleton
        /// </summary>
        public static ILog Instance
        {
            get { return m_log ?? (m_log = new Log()); }
        }
    }
}