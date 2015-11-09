using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    public class CellPhone_YAXNamespaceOverridesImplicitNamespace
    {
        [YAXNamespace("http://namespace.org/explicitBrand")]
        [YAXSerializeAs("{http://namespace.org/implicitBrand}Brand")]
        public string DeviceBrand { get; set; }

        [YAXSerializeAs("{http://namespace.org/os}OperatingSystem")]
        public string OS { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CellPhone_YAXNamespaceOverridesImplicitNamespace GetSampleInstance()
        {
            return new CellPhone_YAXNamespaceOverridesImplicitNamespace
            {
                DeviceBrand = "Samsung Galaxy S II",
                OS = "Android 2",
            };
        }
    }
}
