using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using Troschuetz.Random.Distributions.Continuous;
using World.GameActors.Tiles;

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

        void RegisterHeatSource(IHeatSource heatSource);

        void UnregisterHeatSource(IHeatSource heatSource);
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

        private readonly NormalDistribution m_random = new NormalDistribution(42, 0, 1);

        private readonly List<IHeatSource> m_heatSources;
        private const float MAX_AFFECTED_DISTANCE = 10f;

        public Atmosphere(IAtlas atlas)
        {
            m_atlas = atlas;
            m_heatSources = new List<IHeatSource>();
            m_oldTemperature = 1f;
            

            DateTime time = atlas.Time;
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
            TimeSpan timeSpan = m_atlas.Time - m_prevCycleStart;
            int seconds = (int) timeSpan.TotalSeconds;

            float cyclePhase = MathHelper.Pi*seconds/SECONDS_IN_12_HOURS + MathHelper.Pi;
            float increase = (1f + (float) Math.Cos(cyclePhase))/2;
            float weatherTemperature = m_oldTemperature + increase*m_newDiff;

            float innerTemperature = 0;
            string actualRoomName = m_atlas.AreasCarrier.RoomName(position);
            IEnumerable<IHeatSource> heatSources = m_heatSources.Where(
                x => Vector2.Distance(position, (Vector2) x.Position) < MAX_AFFECTED_DISTANCE);
            // WolframAlpha.com: Plot(a-(11 a x)/60+(a x^2)/120, {x,0,10},{a,0,10})
            foreach (IHeatSource source in heatSources)
            {
                string sourceRoomName = m_atlas.AreasCarrier.RoomName((Vector2)source.Position);
                bool inSameRoom = sourceRoomName == actualRoomName;
                if(!inSameRoom) continue;
                float distance = Vector2.Distance(Tile.Center(source.Position), position);
                float heat = source.Heat;
                innerTemperature += heat - (11f*heat*distance)/60f + (heat*distance*distance)/120f;
            }

            return weatherTemperature + innerTemperature;
        }

        public void Update()
        {
            DateTime actualTime = m_atlas.Time;
            TimeSpan span = actualTime - m_prevCycleStart;
            if (span.TotalSeconds < SECONDS_IN_12_HOURS) return;

            float winterPart = Math.Abs((actualTime.Month - 6.5f)/5.5f);
            m_dailyTemperature = SUMMER_MID_TEMPERATURE + winterPart*-SUMMER_WINTER_DIFF;

            bool getCold = actualTime.Hour >= 18;
            float newTemperature = m_oldTemperature + m_newDiff;
            float newDiff = Math.Abs(newTemperature - m_dailyTemperature)*(float) (m_random.NextDouble()) / 10 + 0.2f;
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

        public void RegisterHeatSource(IHeatSource heatSource)
        {
            m_heatSources.Add(heatSource);
        }

        public void UnregisterHeatSource(IHeatSource heatSource)
        {
            m_heatSources.Remove(heatSource);
        }
    }
}