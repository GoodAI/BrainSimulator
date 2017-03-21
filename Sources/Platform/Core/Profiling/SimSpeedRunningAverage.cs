using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Platform.Core.Profiling
{
    public class SimSpeedRunningAverage
    {
        private const int IntervalCount = 10;
        private readonly long m_minRecordIntervalTicks;

        private struct TimePoint
        {
            public void Set(long steps, long ticks)
            {
                Steps = steps;
                Ticks = ticks;
            }

            public long Steps { get; private set; }
            public long Ticks { get; private set; }
        }

        private readonly TimePoint[] m_timePoints = new TimePoint[IntervalCount];

        private readonly Stopwatch m_stopwatch = Stopwatch.StartNew();

        private int m_index;

        private long m_lastElapsedTicks;

        public SimSpeedRunningAverage(long minRecordIntervalMillisec = 0)
        {
            m_minRecordIntervalTicks = minRecordIntervalMillisec * Stopwatch.Frequency / 1000;
        }

        public void Restart()
        {
            m_stopwatch.Restart();
            m_index = 0;
        }

        public void AddTimePoint(long stepCount, bool force = false)
        {
            if (!force && (m_minRecordIntervalTicks > 0)
                && (m_stopwatch.ElapsedTicks - m_lastElapsedTicks < m_minRecordIntervalTicks))
            {
                return;
            }

            m_lastElapsedTicks = m_stopwatch.ElapsedTicks;

            m_timePoints[m_index % IntervalCount].Set(stepCount, m_stopwatch.ElapsedTicks);

            m_index++;
        }

        public float GetItersPerSecond()
        {
            if (m_index < 2)
                return 0.0f;

            var oldestTimePoint = m_timePoints[Math.Max(m_index - IntervalCount, 0) % IntervalCount];

            var newestTimePoint = m_timePoints[(m_index - 1) % IntervalCount];

            var stepCount = newestTimePoint.Steps - oldestTimePoint.Steps;
            if (stepCount < 0)
                throw new InvalidOperationException("New step count is smaller then old one.");

            var ticks = newestTimePoint.Ticks - oldestTimePoint.Ticks;
            if (ticks < 0)
                throw new InvalidOperationException("New elapsed ticks is lower then old ones.");

            if (ticks == 0)
                return 0.0f;

            return (float)stepCount * Stopwatch.Frequency / ticks;
        }
    }
}
