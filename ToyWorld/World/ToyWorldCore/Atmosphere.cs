using System;
using VRageMath;
using Troschuetz.Random.Distributions.Continuous;

namespace World.ToyWorldCore
{
    public interface IAtmosphere
    {
        /// <summary>
        /// Temperature at given position.
        /// </summary>
        /// <param name="position"></param>
        /// <returns></returns>
        float Temperature(Vector2 position);

        /// <summary>
        /// Call before Temperature() call or every step.
        /// </summary>
        void Update();
    }

    public class Atmosphere : IAtmosphere
    {
        private readonly IAtlas m_atlas;

        private float m_oldTemperature;
        private float m_newDiff;
        private DateTime m_prevCycleStart;

        private const int SECONDS_IN_12_HOURS = 60 * 60 * 12;
        private const long TICKS_IN_12_HOURS = SECONDS_IN_12_HOURS * 10000000L;

        private const float SUMMER_MID_TEMPERATURE = 1.5f;
        private const float SUMMER_WINTER_DIFF = 1f;

        private float m_dailyTemperature = 1f;

        private readonly NormalDistribution m_random = new NormalDistribution(42);

        public Atmosphere(IAtlas atlas)
        {
            m_atlas = atlas;
            m_oldTemperature = 1f;
            

            DateTime time = atlas.Time();
            bool getCold = time.Hour < 6 || time.Hour >= 18;
            if (getCold)
            {
                m_newDiff = -0.1f;
            }
            else
            {
                m_newDiff = 0.1f;
            }

            long cycles = time.Ticks / TICKS_IN_12_HOURS;
            m_prevCycleStart = new DateTime(cycles * TICKS_IN_12_HOURS - TICKS_IN_12_HOURS / 2);

        }

        public float Temperature(Vector2 position)
        {
            TimeSpan timeSpan = m_atlas.Time() - m_prevCycleStart;
            int seconds = (int)timeSpan.TotalSeconds;

            float cyclePhaze = MathHelper.Pi*seconds/SECONDS_IN_12_HOURS + MathHelper.Pi;
            float increase = (1f + (float)Math.Cos(cyclePhaze)) / 2;
            return m_oldTemperature + increase*m_newDiff;
        }

        public void Update()
        {
            DateTime actualTime = m_atlas.Time();
            TimeSpan span = actualTime - m_prevCycleStart;
            if (span.TotalSeconds < SECONDS_IN_12_HOURS) return;

            float winterPart = Math.Abs((actualTime.Month - 6.5f)/5.5f);
            m_dailyTemperature = SUMMER_MID_TEMPERATURE + winterPart*-SUMMER_WINTER_DIFF;

            bool getCold = actualTime.Hour >= 18;

            float newTemperature = m_oldTemperature + m_newDiff;
            float newDiff = Math.Abs(newTemperature - m_dailyTemperature)*(float) (m_random.NextDouble() - 1) / 10 + 0.2f;
            if (getCold)
            {
                m_newDiff = - newDiff;
            }
            else
            {
                m_newDiff = + newDiff;
            }
            m_oldTemperature = newTemperature;
            m_prevCycleStart  = m_prevCycleStart.Add(new TimeSpan(TICKS_IN_12_HOURS));
        }
    }
}