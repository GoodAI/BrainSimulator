using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]
    [YAXComment(@"This example shows serialization of multi-dimensional, 
        and jagged arrays")]
    public class MultiDimArraySample
    {
        public int[,] IntArray { get; set; }

        public double[, ,] DoubleArray { get; set; }

        public int[][] JaggedArray { get; set; }

        [YAXComment("The containing element should not disappear because of the dims attribute")]
        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement)]
        public int[,] IntArrayNoContainingElems { get; set; }

        [YAXComment("This element should not be serialized serially because each element is not of basic type")]
        [YAXCollection(YAXCollectionSerializationTypes.Serially)]
        public int[][] JaggedNotSerially { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static MultiDimArraySample GetSampleInstance()
        {
            int[,] intArray = new int[2,3];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    intArray[i, j] = i + j + 1;


            double[, ,] doubleArray = new double[2, 3, 3];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        doubleArray[i, j, k] = (double)(i * j + 1) / (k + 0.5);

            int[][] jaggedArray = new int[3][];
            for (int i = 0; i < 3; i++)
            {
                jaggedArray[i] = new int[(i + 1) * 2];
                for (int j = 1; j <= (i + 1) * 2; j++)
                    jaggedArray[i][j - 1] = j;
            }

            return new MultiDimArraySample()
            {
                IntArray = intArray,
                DoubleArray = doubleArray,
                JaggedArray = jaggedArray,
                IntArrayNoContainingElems = intArray,
                JaggedNotSerially = jaggedArray
            };
        }
    }
}
