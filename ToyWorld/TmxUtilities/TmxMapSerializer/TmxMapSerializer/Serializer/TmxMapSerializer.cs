using System.Xml.Serialization;
using TmxMapSerializer.Elements;

namespace TmxMapSerializer.Serializer
{
    public class TmxSerializer : XmlSerializer
    {
        public TmxSerializer() : base(typeof (Map))
        {
        }
    }
}
