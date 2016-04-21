using System;
using GoodAI.Logging;

namespace GoodAI.ToyWorldAPI.Logger
{
    public interface ILoggerDequeuer
    {
        /// <summary>
        /// Returns the oldest message.
        /// </summary>
        /// <returns></returns>
        Tuple<string, object[], Exception> Deqeue();
    }
}