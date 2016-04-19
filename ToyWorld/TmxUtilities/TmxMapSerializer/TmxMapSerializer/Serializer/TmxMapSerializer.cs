using System.IO;
using System.Xml.Serialization;
using TmxMapSerializer.Elements;

namespace TmxMapSerializer.Serializer
{
    public class TmxSerializer : XmlSerializer
    {
        public TmxSerializer() : base(typeof (Map))
        {
        }

        public new Map Deserialize(Stream stream)
        {
            return (Map)base.Deserialize(stream);
        }
    }
}
