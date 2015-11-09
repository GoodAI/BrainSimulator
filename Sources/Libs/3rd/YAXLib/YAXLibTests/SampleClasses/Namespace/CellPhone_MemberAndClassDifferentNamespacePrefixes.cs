using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXNamespace("xmain", "http://namespace.org/nsmain")]
    public class CellPhone_MemberAndClassDifferentNamespacePrefixes
    {
        [YAXSerializeAs("TheName")]
        [YAXNamespace("x1", "http://namespace.org/x1")]
        public string DeviceBrand { get; set; }

        public string OS { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CellPhone_MemberAndClassDifferentNamespacePrefixes GetSampleInstance()
        {
            return new CellPhone_MemberAndClassDifferentNamespacePrefixes
            {
                DeviceBrand = "HTC",
                OS = "Windows Phone 8",
            };
        }

    }
}
