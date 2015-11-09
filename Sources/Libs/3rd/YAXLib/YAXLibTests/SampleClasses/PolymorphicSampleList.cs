using System.Collections.Generic;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXSerializeAs("sample")]
    public abstract class PolymorphicSample
    {
    }

    public class PolymorphicOneSample : PolymorphicSample
    {
    }

    public class PolymorphicTwoSample : PolymorphicSample
    {
    }

    [YAXSerializeAs("samples")]
    public class PolymorphicSampleList : List<PolymorphicSample>
    {
        public static PolymorphicSampleList GetSampleInstance()
        {
            var samples = new PolymorphicSampleList
            {
                new PolymorphicOneSample(),
                new PolymorphicTwoSample(),
            };
            return samples;
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }

    public class PolymorphicSampleListAsMember
    {
        public PolymorphicSampleList SampleList { get; set; }

        public static PolymorphicSampleListAsMember GetSampleInstance()
        {
            return new PolymorphicSampleListAsMember
                       {
                           SampleList = PolymorphicSampleList.GetSampleInstance()
                       };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
    }
}