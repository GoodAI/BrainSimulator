using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

// test class created to discuss:
// http://yaxlib.codeplex.com/discussions/287166
// reported by CodePlex User: GraywizardX

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    public class DictionaryKeyValueAsInterface
    {
        [YAXComment("Values are serialized through a reference to their interface.")]
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
        [YAXDictionary(EachPairName = "attribute", KeyName = "key", SerializeKeyAs = YAXNodeTypes.Attribute)]
        public Dictionary<string, IParameter> Attributes1 { get; set; }

        [YAXComment("Keys are serialized through a reference to their interface.")]
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
        [YAXDictionary(EachPairName = "Entry", ValueName = "value", SerializeValueAs = YAXNodeTypes.Attribute)]
        public Dictionary<IParameter, string> Attributes2 { get; set; }


        public DictionaryKeyValueAsInterface()
        {
            Attributes1 = new Dictionary<string, IParameter>();
            Attributes2 = new Dictionary<IParameter, string>();
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static DictionaryKeyValueAsInterface GetSampleInstance()
        {
            var test = new DictionaryKeyValueAsInterface();

            test.Attributes1.Add("test", new GenericMessageParameter { Name = "name1", Type = "int", Body = "27" });
            test.Attributes2.Add(new GenericMessageParameter { Name = "name2", Type = "str", Body = "30" }, "test");

            return test;
        }
    }

    public interface IParameter
    {
        string Name { get; set; }
        string Type { get; set; }
        string Body { get; set; }
    }

    [YAXSerializeAs("parameter")]
    public abstract class ParameterBase : IParameter
    {
        [YAXSerializeAs("name")]
        [YAXAttributeFor("..")]
        [YAXErrorIfMissed(YAXExceptionTypes.Error)]
        public string Name { get; set; }

        [YAXSerializeAs("type")]
        [YAXAttributeFor("..")]
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
        public string Type { get; set; }

        [YAXValueFor("..")]
        [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
        public string Body { get; set; }
    }

    public class GenericMessageParameter : ParameterBase
    {
    }

}
