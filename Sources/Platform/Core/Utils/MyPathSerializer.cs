using System;
using System.Xml.Linq;
using YAXLib;

namespace GoodAI.Core.Utils
{
    public class MyPathSerializer : ICustomSerializer<string>
    {
        public static string ReferencePath 
        { 
            get { return m_referencePath; }
            set
            {
                m_referencePath = value;

                if (!string.IsNullOrEmpty(m_referencePath))
                {
                    m_referencePath = value.EndsWith("\\") ? m_referencePath : m_referencePath + "\\";
                    m_referenceUri = new Uri(m_referencePath);
                }
                else
                {
                    m_referenceUri = null;
                }
            }
        }

        static MyPathSerializer()
        {
            ReferencePath = String.Empty;
        }

        private static string m_referencePath;
        private static Uri m_referenceUri;

        public string DeserializeFromAttribute(XAttribute attrib)
        {
            return ConvertPathToAbsolute(attrib.Value);
        }

        public string DeserializeFromElement(XElement element)
        {
            return ConvertPathToAbsolute(element.Value);
        }

        public string DeserializeFromValue(string value)
        {
            return ConvertPathToAbsolute(value);
        }

        public void SerializeToAttribute(string objectToSerialize, XAttribute attrToFill)
        {
            attrToFill.Value = ConvertPathToRelative(objectToSerialize);
        }

        public void SerializeToElement(string objectToSerialize, XElement elemToFill)
        {
            elemToFill.Value = ConvertPathToRelative(objectToSerialize);
        }

        public string SerializeToValue(string objectToSerialize)
        {
            return ConvertPathToRelative(objectToSerialize);
        }

        private static string ConvertPathToRelative(string absolutePath)
        {
            if (!string.IsNullOrEmpty(ReferencePath) && !string.IsNullOrEmpty(absolutePath))
            {
                Uri uri = new Uri(absolutePath, UriKind.RelativeOrAbsolute);

                if (uri.IsAbsoluteUri)
                {
                    Uri relativeUri = m_referenceUri.MakeRelativeUri(uri);
                    return relativeUri.OriginalString.Replace('/', '\\');
                }
            }

            return absolutePath;
        }

        private static string ConvertPathToAbsolute(string relativePath)
        {
            if (!string.IsNullOrEmpty(ReferencePath) && !string.IsNullOrEmpty(relativePath))
            {
                Uri uri = new Uri(relativePath, UriKind.RelativeOrAbsolute);

                if (!uri.IsAbsoluteUri)
                {
                    return ReferencePath + Uri.UnescapeDataString(relativePath);
                }
            }

            return relativePath;
        }
    }
}
