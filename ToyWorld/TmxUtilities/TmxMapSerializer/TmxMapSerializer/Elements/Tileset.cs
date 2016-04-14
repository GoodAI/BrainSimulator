using System.Collections.Generic;
using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class Tileset
    {
        [XmlAttribute("source")]
        public string Source { get; set; }

        [XmlAttribute("firstgid")]
        public string Firstgid { get; set; }

        [XmlAttribute("name")]
        public string Name { get; set; }

        [XmlAttribute("tilewidth")]
        public string Tilewidth { get; set; }

        [XmlAttribute("tileheight")]
        public string Tileheight { get; set; }

        [XmlAttribute("spacing")]
        public string Spacing { get; set; }

        [XmlAttribute("tilecount")]
        public string Tilecount { get; set; }

        [XmlAttribute("columns")]
        public string Columns { get; set; }

        [XmlElement("image")]
        public Image Image { get; set; }

        // tileset composed of images

        [XmlElement("tile")]
        public List<TmxTile> Tile { get; set; }
    }
}