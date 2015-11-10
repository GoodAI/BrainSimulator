using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXSerializeAs("MobilePhone")]
    public class CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement
    {
        public string DeviceBrand { get; set; }
        public string OS { get; set; }

        [YAXNamespace("app", "http://namespace.org/apps")]
        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement)]
        public List<string> IntalledApps { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement GetSampleInstance()
        {
            return new CellPhone_CollectionNamespaceGoesThruRecursiveNoContainingElement
            {
                DeviceBrand = "Samsung Galaxy Nexus",
                OS = "Android",
                IntalledApps = new List<string> { "Google Map", "Google+", "Google Play" }
            };
        }
    }
}
