using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Xml.Serialization;
using VRageMath;

namespace TmxMapSerializer.Elements
{
    public class TmxObject
    {
        private bool m_invisible;

        [XmlAttribute("id")]
        public int Id { get; set; }

        [XmlAttribute("name")]
        public string Name { get; set; }

        [XmlAttribute("type")]
        public string Type { get; set; }

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

        [XmlAttribute("rotation")]
        public float Rotation { get; set; }

        [XmlAttribute("visible")]
        public bool Invisible
        {
            get { return m_invisible; }
            set { m_invisible = !value; }
        }

        [XmlElement("properties")]
        public Properties Properties { get; set; }

        [XmlElement("polyline")]
        public Polyline Polyline { get; set; }
    }
}