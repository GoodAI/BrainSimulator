using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace TmxMapSerializer.Elements
{
    public class Properties
    {
        [XmlElement("property")]
        public List<Property> PropertiesList { get; set; }
    }
}
