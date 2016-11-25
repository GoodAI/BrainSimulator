using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class Data
    {
        [XmlAttribute("encoding")]
        public string Encoding { get; set; }

        [XmlText]
        public string RawData { get; set; }
    }
}