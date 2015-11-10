using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]
    public class DictionaryKeyValueAsContent
    {
        [YAXDictionary(EachPairName = "Pair", KeyName = "Digits", ValueName = "Letters",
            SerializeKeyAs=YAXNodeTypes.Attribute, SerializeValueAs = YAXNodeTypes.Content)]
        public Dictionary<int, string> DicValueAsContent { get; set; }

        [YAXDictionary(EachPairName = "Pair", KeyName = "Digits", ValueName = "Letters",
            SerializeKeyAs = YAXNodeTypes.Content, SerializeValueAs = YAXNodeTypes.Attribute)]
        public Dictionary<int, string> DicKeyAsContnet { get; set; }

        [YAXDictionary(EachPairName = "Pair", KeyName = "Digits", ValueName = "Letters",
            SerializeKeyAs = YAXNodeTypes.Content, SerializeValueAs = YAXNodeTypes.Element)]
        public Dictionary<int, string> DicKeyAsContentValueAsElement { get; set; }

        [YAXDictionary(EachPairName = "Pair", KeyName = "Digits", ValueName = "Letters",
            SerializeKeyAs = YAXNodeTypes.Element, SerializeValueAs = YAXNodeTypes.Content)]
        public Dictionary<int, string> DicValueAsContentKeyAsElement { get; set; }

        public static DictionaryKeyValueAsContent GetSampleInstance()
        {
            var dic = new Dictionary<int, string>() 
                { {1, "one"}, {2, "two"}, {3, "three"} };

            return new DictionaryKeyValueAsContent()
            {
                DicValueAsContent = dic,
                DicKeyAsContnet = dic,
                DicKeyAsContentValueAsElement = dic,
                DicValueAsContentKeyAsElement = dic
            };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }
}
