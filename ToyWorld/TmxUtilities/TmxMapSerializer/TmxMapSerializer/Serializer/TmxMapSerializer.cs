using System.Xml.Serialization;
using TmxMapSerializer.Elements;

namespace TmxMapSerializer.Serializer
{
    public class TmxMapSerializer : XmlSerializer
    {
        public TmxMapSerializer() : base(typeof (Map))
        {
        }
    }
}
