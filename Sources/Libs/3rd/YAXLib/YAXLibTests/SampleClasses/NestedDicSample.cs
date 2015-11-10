using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment(@"This example demonstrates usage of recursive collection serialization
                and deserialization. In this case a Dictionary whose Key, or Value is 
                another dictionary or collection has been used.")]
    public class NestedDicSample
    {
        public Dictionary<Dictionary<double, Dictionary<int, int>>, Dictionary<Dictionary<string, string>, List<double>>> SomeDic { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static NestedDicSample GetSampleInstance()
        {
            var dicKV1 = new Dictionary<int, int>();
            dicKV1.Add(1, 1);
            dicKV1.Add(2, 2);
            dicKV1.Add(3, 3);
            dicKV1.Add(4, 4);

            var dicKV2 = new Dictionary<int, int>();
            dicKV2.Add(9, 1);
            dicKV2.Add(8, 2);

            var dicVK1 = new Dictionary<string, string>();
            dicVK1.Add("Test", "123");
            dicVK1.Add("Test2", "456");

            var dicVK2 = new Dictionary<string, string>();
            dicVK2.Add("Num1", "123");
            dicVK2.Add("Num2", "456");

            var dicK = new Dictionary<double, Dictionary<int, int>>();
            dicK.Add(0.99999, dicKV1);
            dicK.Add(3.14, dicKV2);

            var dicV = new Dictionary<Dictionary<string, string>, List<double>>();
            dicV.Add(dicVK1, new double[] { 0.98767, 232, 13.124}.ToList());
            dicV.Add(dicVK2, new double[] { 9.8767, 23.2, 1.34 }.ToList());

            var mainDic = new Dictionary<Dictionary<double, Dictionary<int, int>>, Dictionary<Dictionary<string, string>, List<double>>>();
            mainDic.Add(dicK, dicV);

            return new NestedDicSample()
            {
                SomeDic = mainDic
            };
        }
    }
}
