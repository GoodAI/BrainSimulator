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

        public new Map Deserialize(TextReader textReader)
        {
            return (Map) base.Deserialize(textReader);
        }
    }
}
