using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication(SortKey="002")]

    [YAXComment("This example demonstrates serailizing a very simple struct")]
    public struct BookStruct
    {
        public string Title { get; set; }
        public string Author { get; set; }
        public int PublishYear { get; set; }
        public double Price { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static BookStruct GetSampleInstance()
        {
            return new BookStruct()
            {
                Title = "Reinforcement Learning an Introduction",
                Author = "R. S. Sutton & A. G. Barto",
                PublishYear = 1998,
                Price = 38.75
            };
        }
    }
}
