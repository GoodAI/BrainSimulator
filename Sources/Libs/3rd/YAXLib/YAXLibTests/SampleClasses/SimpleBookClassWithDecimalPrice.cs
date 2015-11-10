// This class is created to test serialization of decimal fields.
// The patch is contributed by 

using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXComment("This example demonstrates serailizing a very simple class")]
    public class SimpleBookClassWithDecimalPrice
    {
        public string Title { get; set; }
        public string Author { get; set; }
        public int PublishYear { get; set; }
        public decimal Price { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static SimpleBookClassWithDecimalPrice GetSampleInstance()
        {
            return new SimpleBookClassWithDecimalPrice()
            {
                Title = "Inside C#",
                Author = "Tom Archer & Andrew Whitechapel",
                PublishYear = 2002,
                Price = 32.20m //30.5
            };
        }
    }
}
