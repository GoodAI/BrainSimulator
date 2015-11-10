using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses.Namespace
{
    [YAXNamespace("http://namespace.org/default")]
    public class MutlilevelObjectsWithNamespaces
    {
        public Class1Parent Parent1 { get; set; }
        public Class2Parent Parent2 { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static MutlilevelObjectsWithNamespaces GetSampleInstance()
        {
            var parent1 = new Class1Parent
            {
                Child1 = new Class1Child 
                { 
                    Field1 = "Field1",
                    Field2 = "Field2"
                }
            };

            var parent2 = new Class2Parent
            {
                Child2 = new Class2Child
                {
                    Value1 = "Value1",
                    Value2 = "Value2",
                    Value3 = "Value3",
                    Value4 = "Value4"

                }
            };

            return new MutlilevelObjectsWithNamespaces
            {
                Parent1 = parent1,
                Parent2 = parent2
            };
        }
    }

    public class Class1Parent
    {
        [YAXNamespace("ch1", "http://namespace.org/ch1")]
        public Class1Child Child1 { get; set; }
    }

    public class Class1Child
    {
        public string Field1 { get; set; }
        public string Field2 { get; set; }
    }

    public class Class2Parent
    {
        [YAXNamespace("ch2", "http://namespace.org/ch2")]
        public Class2Child Child2 { get; set; }
    }

    public class Class2Child
    {
        [YAXElementFor("../../Parent1/{http://namespace.org/ch1}Child1")]
        public string Value1 { get; set; }

        [YAXElementFor("../../Parent1/Child1")]
        public string Value2 { get; set; }

        [YAXAttributeFor("../../Parent1/{http://namespace.org/ch1}Child1")]
        public string Value3 { get; set; }

        public string Value4 { get; set; }
    }

}
