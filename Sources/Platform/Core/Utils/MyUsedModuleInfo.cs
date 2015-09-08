using System.Collections.Generic;
using System.Xml.Linq;
using System.Xml.XPath;

using YAXLib;

namespace GoodAI.Core.Utils
{
    [YAXSerializeAs("Module"), YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class MyUsedModuleInfo
    {
        [YAXAttributeForClass]
        public string Name { get; set; }
        [YAXAttributeForClass, YAXSerializableField(DefaultValue = 1)]
        public int Version { get; set; }

        public static List<MyUsedModuleInfo> DeserializeUsedModulesSection(string xml)
        {
            List<MyUsedModuleInfo> result = new List<MyUsedModuleInfo>();

            XDocument doc = XDocument.Parse(xml);

            IEnumerable<XElement> modules = doc.XPathSelectElements("//UsedModules/Module");

            foreach (XElement moduleElm in modules)
            {

                XAttribute nameAttrib = moduleElm.Attribute("Name");
                XAttribute versionAttrib = moduleElm.Attribute("Version");
                int version = 1;

                if (nameAttrib != null && versionAttrib != null && int.TryParse(versionAttrib.Value, out version))
                {
                    result.Add(new MyUsedModuleInfo() { Name = nameAttrib.Value, Version = int.Parse(versionAttrib.Value) });
                }
                else
                {
                    MyLog.ERROR.WriteLine("UsedModules section corrupted at: " + moduleElm.ToString());
                }
            }

            return result;
        }
    }
}
