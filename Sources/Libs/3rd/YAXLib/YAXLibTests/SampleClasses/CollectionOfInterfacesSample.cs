using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public interface ISample
    {
        int IntInInterface { get; set; }
    }

    public class Class1 : ISample
    {
        public int IntInInterface { get; set; }
        public double DoubleInClass1 { get; set; }
    }

    public class Class2 : ISample
    {
        public int IntInInterface { get; set; }
        public string StringInClass2 { get; set; }
    }

    public class Class3_1 : Class1
    {
        public string StringInClass3_1 { get; set; }
    }

    [ShowInDemoApplication]
    [YAXComment(@"This example shows serialization and deserialization of
        objects through a reference to their base class or interface")]
    public class CollectionOfInterfacesSample
    {
        public ISample SingleRef { get; set; }
        public List<ISample> ListOfSamples { get; set; }
        public Dictionary<ISample, int> DictSample2Int { get; set; }
        public Dictionary<int, ISample> DictInt2Sample { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static CollectionOfInterfacesSample GetSampleInstance()
        {
            var c1 = new Class1() { IntInInterface = 1, DoubleInClass1 = 1.0 };
            var c2 = new Class2() { IntInInterface = 2, StringInClass2 = "Class2" };
            var c3 = new Class3_1() { DoubleInClass1 = 3.0, IntInInterface = 3, StringInClass3_1 = "Class3_1" };

            List<ISample> lstOfSamples = new List<ISample>();
            lstOfSamples.Add(c1);
            lstOfSamples.Add(c2);
            lstOfSamples.Add(c3);

            Dictionary<ISample, int> dicSample2Int = new Dictionary<ISample, int>();
            dicSample2Int.Add(c1, 1);
            dicSample2Int.Add(c2, 2);
            dicSample2Int.Add(c3, 3);

            Dictionary<int, ISample> dicInt2Sample = new Dictionary<int, ISample>();
            dicInt2Sample.Add(1, c1);
            dicInt2Sample.Add(2, c2);
            dicInt2Sample.Add(3, c3);


            return new CollectionOfInterfacesSample()
            {
                SingleRef = new Class2() { IntInInterface = 22, StringInClass2 = "SingleRef" },
                ListOfSamples = lstOfSamples,
                DictSample2Int = dicSample2Int,
                DictInt2Sample = dicInt2Sample
            };
        }
    }
}
