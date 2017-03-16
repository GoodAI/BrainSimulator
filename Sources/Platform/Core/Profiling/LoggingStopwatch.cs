using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;
using GoodAI.Core;

namespace GoodAI.Platform.Core.Profiling
{
    public class LoggingStopwatch
    {
        private readonly Stopwatch m_stopwatch = new Stopwatch();

        #region inner class TimeSegment

        private class TimeSegment
        {
            public TimeSegment(string title)
            {
                m_title = title;

                Reset();
            }

            private readonly string m_title;

            private const int MaxTitleLength = 25;

            public int IterCount { get; private set; }

            private long m_totalTicks;

            public long MinTicks { get; private set; }
            public long MaxTicks { get; private set; }

            /// <summary>
            /// Returns average number of ticks, but 0 when IterCount == 0 (to prevent div by zero).
            /// </summary>
            public long AvgTicks => (IterCount > 0) ? m_totalTicks / IterCount : 0;

            public void Reset()
            {
                m_totalTicks = 0;

                IterCount = 0;
                MinTicks = long.MaxValue;
                MaxTicks = 0;
            }

            public void AddElapsedTicks(long ticks)
            {
                m_totalTicks += ticks;
                IterCount++;

                MinTicks = Math.Min(MinTicks, ticks);
                MaxTicks = Math.Max(MaxTicks, ticks);
            }

            public void PrintInfo(string title = null)
            {
                title = title ?? m_title;

                if (title.Length > MaxTitleLength)
                {
                    title = title.Substring(0, MaxTitleLength - 1) + "~";
                }

                title = "  " + title.PadRight(MaxTitleLength, '·');

                if (IterCount == 0)
                {
                    MyLog.INFO.WriteLine($"{title} No stats collected!");
                    return;
                }

                MyLog.INFO.WriteLine(title
                    + $" min:avg({IterCount}):max[μs] "
                    + PrintMicrosecItem(MinTicks) + " : "
                    + PrintMicrosecItem(AvgTicks) + " : "
                    + PrintMicrosecItem(MaxTicks));
            }
        }

        #endregion

        private readonly OrderedDictionary m_segments = new OrderedDictionary(capacity: 16);

        private readonly TimeSegment m_enclosingSegment = new TimeSegment("Total");
        private TimeSegment m_lastSegmentRef;

        private readonly int m_itersPerBatch;
        private readonly string m_title;
        private bool m_isFirstBatch;

        private long m_enclosingSegmentTicks = 0;

        public object ContextId { get; set; }

        private string ContextName => (ContextId.GetHashCode() % 10000).ToString().PadLeft(4);

        public int GpuNo { get; set; }
        public bool SynchronizeGpu { get; set; }

        public LoggingStopwatch(string title = "", int iterationCountPerBatch = 20, bool shouldHideFirstBatch = false)
        {
            m_title = title;
            m_itersPerBatch = iterationCountPerBatch;
            m_isFirstBatch = shouldHideFirstBatch;

            ContextId = this;
        }

        public void Start()
        {
            m_lastSegmentRef = null;
            m_stopwatch.Restart();
        }

        public void StartNewSegment(string key)
        {
            if (SynchronizeGpu)
                MyKernelFactory.Instance.GetContextByGPU(GpuNo).Synchronize();

            if (m_stopwatch.IsRunning)
                CloseSegment();

            if (!m_segments.Contains(key))
            {
                m_segments.Add(key, new TimeSegment(key));
            }

            m_lastSegmentRef = m_segments[key] as TimeSegment;

            m_stopwatch.Restart();
        }

        [Obsolete]
        public void SynchronizeAndStartNewSegment(string key, int gpuNo)
        {
            GpuNo = gpuNo;
            StartNewSegment(key);
        }

        private void CloseSegment()
        {
            var elapsedTicks = m_stopwatch.ElapsedTicks;
            m_lastSegmentRef?.AddElapsedTicks(elapsedTicks);

            m_enclosingSegmentTicks += elapsedTicks;
        }

        public void StopAndSometimesPrintStats()
        {
            m_stopwatch.Stop();

            CloseSegment();

            m_enclosingSegment.AddElapsedTicks(m_enclosingSegmentTicks);
            m_enclosingSegmentTicks = 0;

            // ReSharper disable once InvertIf
            if (m_enclosingSegment.IterCount >= m_itersPerBatch)
            {
                if (!m_isFirstBatch)
                    PrintStats();

                m_isFirstBatch = false;
                RestartBatch();
            }
        }

        private void PrintStats()
        {
            if (m_segments.Count > 1)
            {
                MyLog.INFO.WriteLine($"{m_title}[{ContextName}]:");

                foreach (var segment in m_segments.Values.Cast<TimeSegment>())
                {
                    segment.PrintInfo();
                }
            }

            m_enclosingSegment.PrintInfo((m_segments.Count > 1)
                ? "Total"
                : $"{m_title}[{ContextName}] ");

        }

        private void RestartBatch()
        {
            foreach (var segment in m_segments.Values.Cast<TimeSegment>())
            {
                segment.Reset();
            }

            m_enclosingSegment.Reset();
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
