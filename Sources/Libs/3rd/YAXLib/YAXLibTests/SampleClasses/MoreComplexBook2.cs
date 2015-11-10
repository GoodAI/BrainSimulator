using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    public class Author
    {
        [YAXSerializeAs("Author's Name")]
        [YAXAttributeFor("..")]
        public string Name { get; set; }

        [YAXSerializeAs("Author's Age")]
        [YAXElementFor("../Something/Or/Another")]
        public int Age { get; set; }
    }

    [ShowInDemoApplication]
    [YAXComment(@"This class shows how members of nested objects 
        can be serialized in their parents using serialization 
        addresses including ""..""")]
    public class MoreComplexBook2
    {
        public string Title { get; set; }

        public Author Author { get; set; }

        public int PublishYear { get; set; }
        public double Price { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static MoreComplexBook2 GetSampleInstance()
        {
            Author auth = new Author() { Age = 30, Name = "Tom Archer" };
            return new MoreComplexBook2()
            {
                Title = "Inside C#",
                Author = auth,
                PublishYear = 2002,
                Price = 30.5
            };
        }
    }
}
