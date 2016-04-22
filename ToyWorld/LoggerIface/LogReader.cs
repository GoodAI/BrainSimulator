using System;
using Logger;

namespace GoodAI.Logging
{
    /// <summary>
    /// Logger. Call Log.Instance.Method(). Implemented as queue (FIFO).
    /// </summary>
    public class LogReader : Log, ILogReader
    {
        private static LogReader m_log;

        public LogMessage GetNextLogMessage()
        {
            LogMessage message = null;
            Queue.TryDequeue(out message);
            return message;
        }
        /// <summary>
        /// Singleton
        /// </summary>
        public static new ILogReader Instance
        {
            get { return m_log ?? (m_log = new LogReader()); }
        }
    }
}
