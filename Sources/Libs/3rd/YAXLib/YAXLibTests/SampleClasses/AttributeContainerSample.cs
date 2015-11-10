using System.Collections;
using System.Collections.Generic;

using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXSerializeAs("container")]
    public class AttributeContainerSample
    {
        [YAXSerializeAs("range")]
        public AttributeSample Range { get; set; } 

        public static AttributeContainerSample GetSampleInstance()
        {
            var container = new AttributeContainerSample
            {
                Range = new AttributeSample
                {
                    From = 1,
                    To = 3,
                }
            };

            return container;
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

    public class AttributeSample
    {
        [YAXSerializeAs("from")]
        [YAXAttributeForClass]
        public int? From { get; set; }

        [YAXSerializeAs("to")]
        [YAXAttributeForClass]
        public int? To { get; set; }
    }

    public interface IAttributeSample<T> : IList<T>
    {
        string Url { get; set; }
        int Page { get; }
    }

    public abstract class AttributeSampleBase<T> : List<T>, IAttributeSample<T>
    {
        [YAXSerializeAs("url")]
        [YAXAttributeForClass]
        public string Url { get; set; }

        [YAXSerializeAs("page")]
        [YAXAttributeForClass]
        public int Page
        {
            get { return 1; }
        }
    }

    [YAXSerializeAs("subclass")]
    public class AttributeSubclassSample : AttributeSampleBase<AttributeSample>
    {
        public static AttributeSubclassSample GetSampleInstance()
        {
            var instance = new AttributeSubclassSample
            {
                Url = "http://example.com/subclass/1",
            };

            //instance.Add(new AttributeSample { From = 1, To = 2 });
            //instance.Add(new AttributeSample { From = 3, To = 4 });
            //instance.Add(new AttributeSample { From = 5, To = 6 });

            return instance;
        }
    }
}
