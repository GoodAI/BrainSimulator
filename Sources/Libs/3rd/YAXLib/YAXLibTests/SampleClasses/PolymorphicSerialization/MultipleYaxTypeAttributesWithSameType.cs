using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.PolymorphicSerialization
{
    public class MultipleYaxTypeAttributesWithSameType
    {
        [YAXType(typeof(string))]
        [YAXType(typeof(string))]
        public object Object { get; set; }
    }
}
