using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    [YAXComment("This exmaple shows the usage of nullable fields")]
    public class NullableClass
    {
        public string Title { get; set; }
        public int PublishYear { get; set; }
        public int? PurchaseYear { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static NullableClass GetSampleInstance()
        {
            return new NullableClass()
            {
                Title = "Inside C#",
                PublishYear = 2002,
                PurchaseYear = null
            };
        }
    }
}
