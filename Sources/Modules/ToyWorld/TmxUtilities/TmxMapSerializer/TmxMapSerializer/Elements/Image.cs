using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class Image
    {
        [XmlAttribute("source")]
        public string Source { get; set; }

        [XmlAttribute("trans")]
        public string Trans { get; set; }

        [XmlAttribute("width")]
        public string Width { get; set; }

        [XmlAttribute("height")]
        public string Height { get; set; }
    }
}