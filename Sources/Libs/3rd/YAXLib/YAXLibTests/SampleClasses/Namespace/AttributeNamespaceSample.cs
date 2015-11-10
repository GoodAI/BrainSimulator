using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXNamespace("http://namespaces.org/default")]
    public class AttributeNamespaceSample
    {
        [YAXAttributeFor("Attribs")]
        public string attrib { get; private set; }

        [YAXAttributeFor("Attribs")]
        [YAXNamespace("ns", "http://namespaces.org/ns")]
        public string attrib2 { get; private set; }

        public static AttributeNamespaceSample GetSampleInstance()
        {
            return new AttributeNamespaceSample()
            {
                attrib = "value",
                attrib2 = "value2"
            };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }
}
