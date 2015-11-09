using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("This example shows serialization and deserialization of GUID obejcts")]
    public class GUIDTest
    {
        public Guid StandaloneGuid { get; set; }
        public Dictionary<Guid, int> SomeDic { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static GUIDTest GetSampleInstance()
        {
            return GetSampleInstance(Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid());
        }

        public static GUIDTest GetSampleInstance(Guid g1, Guid g2, Guid g3, Guid g4)
        {
            Dictionary<Guid, int> dic = new Dictionary<Guid, int>();
            dic.Add(g1, 1);
            dic.Add(g2, 2);
            dic.Add(g3, 3);

            return new GUIDTest
            {
                StandaloneGuid = g4,
                SomeDic = dic
            };
        }

    }
}
