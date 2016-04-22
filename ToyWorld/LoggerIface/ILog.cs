using System;
using Logger;

namespace GoodAI.Logging
{
    public interface ILog
    {
        /// <summary>
        /// Adds message to logger. 
        /// </summary>
        /// <param name="severity"></param>
        /// <param name="template"></param>
        /// <param name="objects"></param>
        void Add(Severity severity, string template, params object[] objects);

        /// <summary>
        /// Adds message to logger.
        /// </summary>
        /// <param name="severity"></param>
        /// <param name="ex"></param>
        /// <param name="template"></param>
        /// <param name="objects"></param>
        void Add(Severity severity, Exception ex, string template, params object[] objects);
    }
}
