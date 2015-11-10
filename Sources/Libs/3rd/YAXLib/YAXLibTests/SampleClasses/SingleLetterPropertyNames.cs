using System.Collections.Generic;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    // This sample class has been contributed by CodePlex user [a5r](https://www.codeplex.com/site/users/view/a5r)
    // which provides another test case to make sure the fix for the issue of single letter property names
    // is actually fixed.
    // Here's the related discussion: https://yaxlib.codeplex.com/discussions/538683
    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
    public class SingleLetterPropertyNames
    {
        [YAXSerializableField]
        public LinkedList<TestPoint> TestPoints { get; set; }

        /// <summary>
        /// Initializes a new instance of the SampleClassForTestPoint class.
        /// </summary>
        public SingleLetterPropertyNames()
        {
            TestPoints = new LinkedList<TestPoint>();
            TestPoints.AddLast(new TestPoint { Id = 0, X = 100, Y = 100 });
            TestPoints.AddLast(new TestPoint { Id = 1, X = -100, Y = 150 });
        }

        public static SingleLetterPropertyNames GetSampleInstance()
        {
            return new SingleLetterPropertyNames();
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public class TestPoint
        {
            public int Id { get; set; }

            public double X { get; set; }

            public double Y { get; set; }
        }
    }
}
