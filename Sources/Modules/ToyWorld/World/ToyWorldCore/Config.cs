using System;

namespace World.ToyWorldCore
{
    public interface IConfig
    {
        TimeSpan StepLength { get; }
        TimeSpan YearLength { get; }
        TimeSpan DayLengh { get; }

        int SignalCount { get; }

        int FruitSpawnPeriod { get; }
        int FruitSpawnRange { get; }
        int FruitFirstSpawn { get; }
        int FruitFirstSpawnRange { get; }
        int FruitRotAfter { get; }

        float BedRechargeRate { get; }
    }

    public class CommonConfig : IConfig
    {
        public TimeSpan StepLength { get{ return new TimeSpan(0,0,1,0); } }
        public TimeSpan YearLength{ get { return new TimeSpan(15,0,0,0); } }
        public TimeSpan DayLengh { get { return new TimeSpan(1,0,0,0); } }

        public int SignalCount { get { return 8; } }

        public int FruitSpawnPeriod { get { return 5000; } }
        public int FruitSpawnRange { get { return 1000; } }
        public int FruitFirstSpawn { get { return 500; } }
        public int FruitFirstSpawnRange { get { return 200; } }
        public int FruitRotAfter { get { return 0; } } // no rotting

        public float BedRechargeRate { get { return 0.1f; } }
    }

    public static class TWConfig
    {
        private static readonly IConfig m_instance = new CommonConfig();
        public static IConfig Instance { get { return m_instance; } }
    }
}
