namespace YAXLibTests.SampleClasses.SelfReferencingObjects
{
    public class DirectSelfReferringObject
    {
        public int Data { get; set; }
        public DirectSelfReferringObject Next { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static DirectSelfReferringObject GetSampleInstance()
        {
            var first = new DirectSelfReferringObject {Data = 1};
            var second = new DirectSelfReferringObject {Data = 2};
            first.Next = second;
            // this must be serialized fine, because there's no loop, although the type is a self referring type.
            // However by setting, "second.Next = first;" It should not be serialized any more because it will cause a loop
            
            return first;
        }

        public static DirectSelfReferringObject GetSampleInstanceWithCycle()
        {
            var instance = GetSampleInstance();
            instance.Next.Next = instance;
            return instance;
        }

        public static DirectSelfReferringObject GetSampleInstanceWithSelfCycle()
        {
            var instance = new DirectSelfReferringObject();
            instance.Data = 1;
            instance.Next = instance;
            return instance;
        }
    }
}
