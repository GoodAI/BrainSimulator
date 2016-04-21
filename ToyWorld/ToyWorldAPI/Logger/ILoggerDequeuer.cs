using System;
using GoodAI.Logging;

namespace GoodAI.ToyWorldAPI.Logger
{
    /// <summary>
    /// Allows pulling and removing from logger.
    /// </summary>
    public interface ILoggerDequeuer
    {
        /// <summary>
        /// Returns the oldest message. If there is no message, returns null.
        /// </summary>
        /// <returns></returns>
        Tuple<string, object[], Exception> Deqeue();
    }
}