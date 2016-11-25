using System;
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
            var fileStream = stream as FileStream;

            if (fileStream == null)
            {
                throw new ArgumentException("Map must be deserialized from .tmx FileStream");
            }

            Map map = (Map)base.Deserialize(stream);

            foreach (Tileset tileset in map.Tilesets)
            {
                tileset.Image.Source = fileStream.Name.Substring(0, fileStream.Name.LastIndexOf(@"\", StringComparison.Ordinal))+ @"\" + tileset.Image.Source;
            }

            return map;
        }
    }
}
