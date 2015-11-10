using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXComment(@"This example shows serialization and deserialization of objects
                through a reference to their base class or interface while used in 
                collection classes")]
    public class InterfaceMatchingSample
    {
        [YAXAttributeForClass]
        public int? SomeNumber { get; set; }

        [YAXCollection(YAXCollectionSerializationTypes.Serially)]
        public List<int?> ListOfSamples { get; set; }

        [YAXDictionary(SerializeKeyAs = YAXNodeTypes.Attribute, SerializeValueAs= YAXNodeTypes.Attribute)]
        public Dictionary<double?, int> DictNullable2Int { get; set; }

        [YAXDictionary(SerializeKeyAs = YAXNodeTypes.Attribute, SerializeValueAs = YAXNodeTypes.Attribute)]
        public Dictionary<int, double?> DictInt2Nullable { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static InterfaceMatchingSample GetSampleInstance()
        {
            var lstOfSamples = new List<int?>();
            lstOfSamples.Add(2);
            lstOfSamples.Add(4);
            lstOfSamples.Add(8);

            var dicSample2Int = new Dictionary<double?, int>();
            dicSample2Int.Add(1.0, 1);
            dicSample2Int.Add(2.0, 2);
            dicSample2Int.Add(3.0, 3);

            var dicInt2Sample = new Dictionary<int, double?>();
            dicInt2Sample.Add(1, 1.0);
            dicInt2Sample.Add(2, 2.0);
            dicInt2Sample.Add(3, null);


            return new InterfaceMatchingSample()
            {
                SomeNumber = 10,
                ListOfSamples = lstOfSamples,
                DictNullable2Int = dicSample2Int,
                DictInt2Nullable = dicInt2Sample
            };
        }
    }
}
