using System;

namespace GoodAI.BasicNodes.DyBM
{
    public class MyRandom
    {
        static Random rand = new Random();

        public double NextDouble(float mean, float stdDev)
        {
            //these are uniform(0,1) random doubles
            double u1 = (float)rand.NextDouble();
            double u2 = (float)rand.NextDouble();
            //random normal(0,1)
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            //random normal(mean,stdDev^2)
            return mean + stdDev * randStdNormal; 
        }
    }
}
