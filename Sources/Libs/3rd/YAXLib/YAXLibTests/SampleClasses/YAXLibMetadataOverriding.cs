using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]
    [YAXNamespace("http://namespace.org/sample")]
    public class YAXLibMetadataOverriding
    {
        public int[,] IntArray { get; set; }

        public object Obj { get; set; }


        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static YAXLibMetadataOverriding GetSampleInstance()
        {
            int[,] intArray = new int[2,3];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    intArray[i, j] = i + j + 1;


            return new YAXLibMetadataOverriding()
            {
                IntArray = intArray,
                Obj = "Hello, World!"
            };
        }
    }
}
