using System.Collections.Generic;
using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{

    public class ObjectGroup
    {
        [XmlAttribute("name")]
        public string Name { get; set; }

        [XmlElement("object")]
        public List<TmxObject> TmxMapObjects { get; set; }
    }
}