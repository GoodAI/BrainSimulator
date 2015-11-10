using System;
using System.Collections.Generic;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXNotCollection]
    public class DictionaryWithExtraPropertiesAttributedAsNotCollection : Dictionary<int, string>
    {
        public string Prop1 { get; set; }
        public double Prop2 { get; set; }

        public static DictionaryWithExtraPropertiesAttributedAsNotCollection GetSampleInstance()
        {
            var inst = new DictionaryWithExtraPropertiesAttributedAsNotCollection
                           {
                               Prop1 = "Prop1",
                               Prop2 = 2.234
                           };
            inst.Add(1, "One");
            inst.Add(2, "Two");
            inst.Add(3, "Three");

            return inst;
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this)
                   + String.Format("Prop1: {0}, Prop2: {1}", Prop1, Prop2);
        }

    }
}
