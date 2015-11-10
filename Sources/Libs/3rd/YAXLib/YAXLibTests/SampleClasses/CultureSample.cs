using System;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXComment("This class contains fields that are vulnerable to culture changes!")]
    public class CultureSample
    {
        public double Number1 { get; set; }

        [YAXAttributeForClass]
        public double Number2 { get; set; }

        public double Number3 { get; set; }

        public double[] Numbers { get; set; }

        public decimal Dec1 { get; set; }

        [YAXAttributeForClass]
        public decimal Dec2 { get; set; }

        public DateTime Date1 { get; set; }

        [YAXAttributeForClass]
        public DateTime Date2 { get; set; }

        public static CultureSample GetSampleInstance()
        {
            return new CultureSample
                       {
                           Date1 = new DateTime(2010, 10, 11, 18, 20, 30),
                           Date2 = new DateTime(2011, 9, 20, 4, 10, 30),
                           Dec1 = 192389183919123.18232131m,
                           Dec2 = 19232389.18391912318232131m,
                           Number1 = 123123.1233,
                           Number2 = 32243.67676,
                           Number3 = 21313.123123,
                           Numbers = new [] { 23213.2132, 123.213, 123.23e32}
                       };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }
}
