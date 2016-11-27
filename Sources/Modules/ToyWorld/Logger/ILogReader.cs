using System;
using Logger;

namespace GoodAI.Logging
{
    public interface ILogReader
    {
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        LogMessage GetNextLogMessage();
    }
}
