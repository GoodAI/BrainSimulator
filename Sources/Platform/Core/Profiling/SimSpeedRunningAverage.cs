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
        private const int IntervalCount = 30;
        private readonly long m_minRecordIntervalTicks;


        private struct TimePoint
        {
            public TimePoint(long steps, long ticks)
            {
                Steps = steps;
                Ticks = ticks;
            }

            public long Steps { get; }
            public long Ticks { get; }
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

            m_timePoints[m_index % IntervalCount] = new TimePoint(stepCount, m_stopwatch.ElapsedTicks);

            m_index++;
        }

        public float GetItersPerSecond(long currentStepCount)
        {
            var oldestTimePoint = m_timePoints[Math.Max(m_index - IntervalCount, 0) % IntervalCount];

            var stepCount = currentStepCount - oldestTimePoint.Steps;
            if (stepCount < 0)
                throw new ArgumentException("Current step count is lower then recorded one", nameof(currentStepCount));

            var ticks = m_stopwatch.ElapsedTicks - oldestTimePoint.Ticks;
            if (ticks < 0)
                throw new InvalidOperationException("Elapsed ticks is lower then recorded ones.");

            if (ticks == 0)
                return 0.0f;

            return (float)stepCount * Stopwatch.Frequency / ticks;
        }
    }
}
