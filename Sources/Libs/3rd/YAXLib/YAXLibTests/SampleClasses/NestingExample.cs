using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXSerializeAs("Pricing")]
    public class Request
    {
        public Request()
        { }

        [YAXAttributeForClass()]
        public string id { get; set; }

        [YAXAttributeFor("version")]
        public string major { get; set; }

        [YAXAttributeFor("version")]
        public string minor { get; set; }

        [YAXSerializeAs("value_date")]
        [YAXElementFor("input")]
        public string valueDate { get; set; }

        [YAXSerializeAs("storage_date")]
        [YAXElementFor("input")]
        public string storageDate { get; set; }

        [YAXSerializeAs("user")]
        [YAXElementFor("input")]
        public string user { get; set; }

        //[YAXElementFor("input")]
        //public string skylab_config { get; set; }

        //[YAXElementFor("skylab_config")]
        //public string job { get; set; }

        [YAXElementFor("input")]
        [YAXSerializeAs("skylab_config")]
        public SkyLabConfig Config { get; set; }

        internal static Request GetSampleInstance()
        {
            return new Request()
            {
                id = "123",
                major = "1",
                minor = "0",
                valueDate = "2010-10-5",
                storageDate = "2010-10-5",
                user = "me",
                Config = new SkyLabConfig() { Config = "someconf", Job = "test" }
            };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

    public class SkyLabConfig
    {
        [YAXSerializeAs("SomeString")]
        public string Config { get; set; }

        [YAXSerializeAs("job")]
        public string Job { get; set; }
    }
}
