using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    public class XmlNamespaceElementNames
    {
        [YAXSerializeAs("NoNs")]
        public string WithoutNamespace { get; set; }

        [YAXSerializeAs("{xs}WithNs")]
        public string WithNamespace { get; set; }

        [YAXSerializeAs("{xs}Another")]
        public string AnotherOne { get; set; }

        
        public static XmlNamespaceElementNames GetSampleInstance()
        {
            return new XmlNamespaceElementNames();
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }
}
