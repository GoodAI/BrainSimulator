using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXSerializeAs("MobilePhone")]
    public class CellPhone_CollectionNamespaceForAllItems
    {
        public string DeviceBrand { get; set; }
        public string OS { get; set; }

        [YAXNamespace("app", "http://namespace.org/apps")]
        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement,
            EachElementName = "{http://namespace.org/appName}AppName")]
        public List<string> IntalledApps { get; set; }

        [YAXNamespace("cls", "http://namespace.org/colorCol")]
        [YAXCollection(YAXCollectionSerializationTypes.Recursive,
            EachElementName = "{http://namespace.org/color}TheColor")]
        public List<Color> AvailableColors { get; set; }

        [YAXNamespace("mdls", "http://namespace.org/modelCol")]
        [YAXCollection(YAXCollectionSerializationTypes.Serially,
            EachElementName = "{http://namespace.org/color}TheModel", // should be ignored
            IsWhiteSpaceSeparator=false, SeparateBy=",")]
        public List<String> AvailableModels { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CellPhone_CollectionNamespaceForAllItems GetSampleInstance()
        {
            return new CellPhone_CollectionNamespaceForAllItems
            {
                DeviceBrand = "Samsung Galaxy Nexus",
                OS = "Android",
                IntalledApps = new List<string> { "Google Map", "Google+", "Google Play" },
                AvailableColors = new List<Color> { Color.Red, Color.Black, Color.White },
                AvailableModels = new List<string> { "S1", "MII", "SXi", "NoneSense" }
            };
        }
    }
}
