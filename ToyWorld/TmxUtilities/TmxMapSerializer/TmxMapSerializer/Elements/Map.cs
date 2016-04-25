using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    [XmlRoot("map")]
    public class Map
    {
        [XmlAttribute("width")]
        public int Width { get; set; }

        [XmlAttribute("height")]
        public int Height { get; set; }

        [XmlAttribute("version")]
        public string Version { get; set; }

        [XmlAttribute("orientation")]
        public string Orientation { get; set; }

        [XmlAttribute("renderorder")]
        public string Renderorder { get; set; }

        [XmlAttribute("tilewidth")]
        public int Tilewidth { get; set; }

        [XmlAttribute("tileheight")]
        public int Tileheight { get; set; }

        [XmlAttribute("nextobjectid")]
        public int Nextobjectid { get; set; }

        [XmlElement("tileset")]
        public List<Tileset> Tilesets { get; set; }

        [XmlElement("layer")]
        public List<Layer> Layers { get; set; }

        [XmlElement("objectgroup")]
        public List<ObjectGroup> ObjectGroups { get; set; }
    }
}