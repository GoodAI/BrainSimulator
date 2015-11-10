using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment(@"This example shows how to provide serialization address
        for elements and attributes. Theses addresses resemble those used
        in known file-systems")]
    public class MoreComplexBook
    {
        [YAXAttributeFor("SomeTag/SomeOtherTag/AndSo")]
        public string Title { get; set; }

        [YAXElementFor("SomeTag/SomeOtherTag/AndSo")]
        public string Author { get; set; }

        public int PublishYear { get; set; }
        public double Price { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static MoreComplexBook GetSampleInstance()
        {
            return new MoreComplexBook()
            {
                Title = "Inside C#",
                Author = "Tom Archer & Andrew Whitechapel",
                PublishYear = 2002,
                Price = 30.5
            };
        }
    }
}
