using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.GameActors.Tiles;

namespace World.Atlas
{
    public class SimpleAtmosphere : IAtmosphere
    {
        private readonly IAtlas m_atlas;
        private readonly List<IHeatSource> m_heatSources;

        public SimpleAtmosphere(IAtlas atlas)
        {
            m_atlas = atlas;
            m_heatSources = new List<IHeatSource>();
        }

        public float Temperature(Vector2 position)
        {
            float temperatureBase = (m_atlas.Summer + 1)/2; // [0.5,1]
            float temperatureThisDay = (m_atlas.Day + 1)/2 - 1; // [-0.5,0]

            float innerTemperature = InnerTemperature(position);

            return temperatureBase + temperatureThisDay + innerTemperature;
        }

        private float InnerTemperature(Vector2 position)
        {
            float innerTemperature = 0;
            string actualRoomName = m_atlas.AreasCarrier.RoomName(position);
            IEnumerable<IHeatSource> heatSources = m_heatSources.Where(
                x => Vector2.Distance(position, (Vector2)x.Position) < x.MaxDistance);
            foreach (IHeatSource source in heatSources)
            {
                string sourceRoomName = m_atlas.AreasCarrier.RoomName((Vector2)source.Position);
                bool inSameRoom = sourceRoomName == actualRoomName;
                if (!inSameRoom) continue;
                float distance = Vector2.Distance(Tile.Center(source.Position), position);
                float heat = source.Heat;
                // WolframAlpha.com: Plot(a-(11 a x)/60+(a x^2)/120, {x,0,10},{a,0,10})
                //innerTemperature += heat - (11f*heat*distance)/60f + (heat*distance*distance)/120f;
                // WolframAlpha.com: interpolation[{0,t},{z,0},{z+1,0}]
                float maxDist = source.MaxDistance;
                innerTemperature +=
                    heat
                    - (heat * distance) / maxDist
                    - (heat * distance) / (1 + maxDist)
                    + (heat * distance * distance) / (maxDist * (1 + maxDist));
            }
            return innerTemperature;
        }

        public void Update()
        {
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