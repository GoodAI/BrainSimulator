using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class TmxObject
    {
        [XmlAttribute("id")]
        public int Id { get; set; }

        [XmlAttribute("gid")]
        public int Gid { get; set; }

        [XmlAttribute("x")]
        public float X { get; set; }

        [XmlAttribute("y")]
        public float Y { get; set; }

        [XmlAttribute("width")]
        public float Width { get; set; }

        [XmlAttribute("height")]
        public float Height { get; set; }
    }
}