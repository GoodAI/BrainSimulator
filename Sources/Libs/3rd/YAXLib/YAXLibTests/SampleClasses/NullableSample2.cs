using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class NullableSample2
    {
        [YAXAttributeForClass]
        public int? Number { get; set; }

        [YAXFormat("o")]
        public DateTime? DateTime { get; set; }

        public decimal? Decimal { get; set; }

        public bool? Boolean { get; set; }

        public Seasons? Enum { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static NullableSample2 GetSampleInstance()
        {
            return new NullableSample2
            {
                Number = 10,
                DateTime = new DateTime(624599050212345678, DateTimeKind.Utc),
                Decimal = 1234.56789m,
                Boolean = true,
                Enum = Seasons.Third,
            };
        }
    }
}
