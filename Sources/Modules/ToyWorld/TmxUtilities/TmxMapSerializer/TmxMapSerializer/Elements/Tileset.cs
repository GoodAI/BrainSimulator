using System.Collections.Generic;
using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class Tileset
    {
        [XmlAttribute("source")]
        public string Source { get; set; }

        [XmlAttribute("firstgid")]
        public int Firstgid { get; set; }

        [XmlAttribute("name")]
        public string Name { get; set; }

        [XmlAttribute("tilewidth")]
        public int Tilewidth { get; set; }

        [XmlAttribute("tileheight")]
        public int Tileheight { get; set; }

        [XmlAttribute("spacing")]
        public int Spacing { get; set; }

        [XmlAttribute("tilecount")]
        public int Tilecount { get; set; }

        [XmlAttribute("columns")]
        public int Columns { get; set; }

        [XmlElement("image")]
        public Image Image { get; set; }
    }
}