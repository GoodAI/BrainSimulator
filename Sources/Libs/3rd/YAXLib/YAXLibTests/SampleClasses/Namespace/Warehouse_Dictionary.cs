using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXNamespace("http://www.mywarehouse.com/warehouse/def/v3")]
    public class Warehouse_Dictionary
    {
        [YAXDictionary(EachPairName = "ItemInfo", KeyName = "Item", ValueName = "Count",
                       SerializeKeyAs = YAXNodeTypes.Attribute,
                       SerializeValueAs = YAXNodeTypes.Attribute)]
        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement)]
        [YAXSerializeAs("ItemQuantities")]
        public Dictionary<string, int> ItemQuantitiesDic { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static Warehouse_Dictionary GetSampleInstance()
        {
            return new Warehouse_Dictionary()
            {
                ItemQuantitiesDic = new Dictionary<string, int>() 
                    { {"Item1", 10}, {"Item4", 30}, {"Item2", 20} },
            };
        }
    }
}
