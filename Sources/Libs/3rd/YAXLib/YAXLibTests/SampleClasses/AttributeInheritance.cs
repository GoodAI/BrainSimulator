using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXSerializeAs("Child")]
    public class AttributeInheritance : AttributeInheritanceBase
    {
        [YAXSerializeAs("TheAge")]
        public double Age { get; set; }

        public static AttributeInheritance GetSampleInstance()
        {
            return new AttributeInheritance()
            {
                Name = "John",
                Age = 30.2
            };
        }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }
        
    }

    [YAXSerializeAs("Base")]
    public class AttributeInheritanceBase
    {
        [YAXSerializeAs("TheName")]
        public string Name { get; set; }
    }
}
