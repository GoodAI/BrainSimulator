using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class Layer
    {
        [XmlAttribute("name")]
        public string Name { get; set; }

        [XmlAttribute("width")]
        public int Width { get; set; }

        [XmlAttribute("height")]
        public int Height { get; set; }

        [XmlElement("data")]
        public Data Data { get; set; }
    }
}