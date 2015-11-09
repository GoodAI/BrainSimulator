using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication(SortKey="001")]

    [YAXComment("This example demonstrates serailizing a very simple class")]
    public class Book
    {
        public string Title { get; set; }
        public string Author { get; set; }
        public int PublishYear { get; set; }
        public double Price { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static Book GetSampleInstance()
        {
            return new Book()
            {
                Title = "Inside C#",
                Author = "Tom Archer & Andrew Whitechapel",
                PublishYear = 2002,
                Price = 30.5
            };
        }
    }
}
