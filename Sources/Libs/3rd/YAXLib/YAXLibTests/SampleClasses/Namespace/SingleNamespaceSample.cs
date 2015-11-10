using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXComment("This example shows usage of a custom default namespace")]
    [YAXNamespace("http://namespaces.org/default")]
    public class SingleNamespaceSample
    {
        public static SingleNamespaceSample GetInstance()
        {
            return new SingleNamespaceSample()
            {
                StringItem = "This is a test string",
                IntItem = 10
            };
        }

        public string StringItem
        { get; set; }

        public int IntItem
        { get; set; }
    }
}
