using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;

namespace GoodAI.Platform.Core.Profiling
{
    public class LoggingStopwatch
    {
        private readonly Stopwatch m_stopwatch = new Stopwatch();

        private int m_iterCount;
        private long m_totalTicks;

        private long m_minTicks;
        private long m_maxTicks; 

        private readonly int m_itersPerBatch;
        private readonly string m_message;

        public object ContextId { get; set; }

        private string ContextName => (ContextId.GetHashCode() % 10000).ToString().PadLeft(4);

        public LoggingStopwatch(string message = "", int iterationCountPerBatch = 20)
        {
            m_itersPerBatch = iterationCountPerBatch;
            m_message = message;

            ContextId = this;

            ResetBatch();
        }

        public void Start()
        {
            m_stopwatch.Restart();
        }

        public void StopAndSometimesPrintStats()
        {
            m_stopwatch.Stop();

            m_totalTicks += m_stopwatch.ElapsedTicks;
            m_iterCount++;

            m_minTicks = Math.Min(m_minTicks, m_stopwatch.ElapsedTicks);
            m_maxTicks = Math.Max(m_maxTicks, m_stopwatch.ElapsedTicks);

            if (m_iterCount >= m_itersPerBatch)
            {
                MyLog.INFO.WriteLine($"{m_message}[{ContextName}] "
                    + $"min:avg({m_iterCount}):max[μs] "
                    + PrintMicrosecItem(m_minTicks) + " : "
                    + PrintMicrosecItem(m_totalTicks / m_iterCount) + " : "
                    + PrintMicrosecItem(m_maxTicks));

                ResetBatch();
            }
        }

        private void ResetBatch()
        {
            m_totalTicks = 0;
            m_iterCount = 0;

            m_minTicks = long.MaxValue;
            m_maxTicks = 0;
        }

        private static string PrintMicrosecItem(long ticks)
        {
            return TicksToMicrosec(ticks).ToString().PadLeft(5);
        }

        private static long TicksToMicrosec(long ticks)
        {
            return ticks * 1000L * 1000L / Stopwatch.Frequency;
        }
    }
}
