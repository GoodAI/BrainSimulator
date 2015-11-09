using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("How multi-line comments are serialized as multiple XML comments")]
    public class MultipleCommentsTest
    {
        [YAXComment(@"Using @ quoted style 
                     comments for multiline comments")]
        public int Dummy { get; set; }

        [YAXComment("Comment 1 for member\nComment 2 for member")]
        public int SomeInt { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static MultipleCommentsTest GetSampleInstance()
        {
            return new MultipleCommentsTest() { SomeInt = 10 };
        }
    }
}
