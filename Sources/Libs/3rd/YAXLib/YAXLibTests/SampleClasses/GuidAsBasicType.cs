using System;
using System.Collections.Generic;
using System.Linq;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class GuidAsBasicType
    {
        [YAXAttributeForClass]
        public Guid GuidAsAttr { get; set; }

        public Guid GuidAsElem { get; set; }

        public Guid[] GuidArray { get; set; }

        [YAXCollection(YAXCollectionSerializationTypes.Serially)]
        public Guid[] GuidArraySerially { get; set; }

        public List<Guid> GuidsList { get; set; }

        public Dictionary<Guid, int> DicKeyGuid { get; set; }

        [YAXDictionary(EachPairName = "Pair", KeyName = "TheGuid", SerializeKeyAs = YAXNodeTypes.Attribute)]
        public Dictionary<Guid, int> DicKeyAttrGuid { get; set; }

        public Dictionary<int, Guid> DicValueGuid { get; set; }

        [YAXDictionary(EachPairName = "Pair", ValueName = "TheGuid", SerializeValueAs = YAXNodeTypes.Attribute)]
        public Dictionary<int, Guid> DicValueAttrGuid { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static GuidAsBasicType GetSampleInstance()
        {
            var guids = new []
                            {
                                new Guid("fed92f33-e351-47bd-9018-69c89928329e"),
                                new Guid("042ba99c-b679-4975-ac4d-2fe563a5dc3e"),
                                new Guid("82071c51-ea20-473b-a541-1ebdf8f158d3"),
                                new Guid("81a3478b-5779-451a-b2aa-fbf69bb11424"),
                                new Guid("d626ba2b-a095-4a34-a376-997e5628dfb9"),
                            };

            var dicKey = new Dictionary<Guid, int> {{guids[0], 1}, {guids[1], 2}, {guids[2], 3}};
            var dicValue = new Dictionary<int, Guid> { { 1, guids[0] }, { 2, guids[2]}, { 3, guids[4]} };

            return new GuidAsBasicType
                       {
                            GuidAsAttr = guids[0],
                            GuidAsElem = guids[1],
                            GuidArray = guids,
                            GuidsList = guids.ToList(),
                            GuidArraySerially = guids,
                            DicKeyAttrGuid = dicKey,
                            DicKeyGuid = dicKey,
                            DicValueAttrGuid = dicValue,
                            DicValueGuid = dicValue
                       };
        }
    }
}
