using System.Collections.Generic;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class CollectionWithExtraProperties : List<int>
    {
        public string Property1 { get; set; }
        public double Property2 { get; set; }

        public static CollectionWithExtraProperties GetSampleInstance()
        {
            var instance = new CollectionWithExtraProperties {Property1 = "Property1", Property2 = 1.234};

            instance.Add(1);
            instance.Add(2);
            instance.Add(3);
            instance.Add(4);

            return instance;
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this) +
                string.Format("Property1: {0}, Property2: {1}", Property1, Property2);
        }

    }
}
