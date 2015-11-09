using System.Collections.Generic;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXNotCollection]
    public class CollectionWithExtraPropertiesAttributedAsNotCollection : List<int>
    {
        public string Property1 { get; set; }
        public double Property2 { get; set; }

        public static CollectionWithExtraPropertiesAttributedAsNotCollection GetSampleInstance()
        {
            var instance = new CollectionWithExtraPropertiesAttributedAsNotCollection { Property1 = "Property1", Property2 = 1.234 };

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
