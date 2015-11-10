using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXNamespace("http://namespace.org/nsmain")]
    public class CellPhone_MultiLevelMemberAndClassDifferentNamespaces
    {
        [YAXElementFor("Level1/Level2")]
        [YAXSerializeAs("TheName")]
        [YAXNamespace("x1", "http://namespace.org/x1")]
        public string DeviceBrand { get; set; }

        public string OS { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CellPhone_MultiLevelMemberAndClassDifferentNamespaces GetSampleInstance()
        {
            return new CellPhone_MultiLevelMemberAndClassDifferentNamespaces
            {
                DeviceBrand = "HTC",
                OS = "Windows Phone 8",
            };
        }

    }
}
