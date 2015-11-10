using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXSerializableType(Options = YAXSerializationOptions.DontSerializeNullObjects, FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    [YAXSerializeAs("font")]
    [YAXNamespace("w", "http://example.com/namespace")]
    public class AttributeWithNamespace
    {
        [YAXSerializableField]
        [YAXSerializeAs("{http://example.com/namespace}name")]
        [YAXAttributeForClass]
        public string Name { get; set; }

        public static AttributeWithNamespace GetSampleInstance()
        {
            return new AttributeWithNamespace 
            {
                Name = "Arial"
            };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

    public class AttributeWithNamespaceAsMember
    {
        public AttributeWithNamespace Member { get; set; }

        public static AttributeWithNamespaceAsMember GetSampleInstance()
        {
            return new AttributeWithNamespaceAsMember
                       {
                           Member = AttributeWithNamespace.GetSampleInstance()
                       };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

}

