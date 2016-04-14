using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class TmxTile
    {
        [XmlAttribute("id")]
        public int Id { get; set; }

        [XmlElement("image")]
        public Image Image;
    }
}
