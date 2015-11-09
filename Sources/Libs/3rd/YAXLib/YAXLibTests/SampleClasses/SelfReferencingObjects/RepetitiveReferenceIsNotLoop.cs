namespace YAXLibTests.SampleClasses.SelfReferencingObjects
{
    public class RepetitiveReferenceIsNotLoop
    {
        public SmallClass RefA { get; set; }
        public SmallClass RefB { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static RepetitiveReferenceIsNotLoop GetSampleInstance()
        {
            var singleRef = new SmallClass {N = 10};
            return new RepetitiveReferenceIsNotLoop
                   {
                       RefA = singleRef,
                       RefB = singleRef
                   };
        }
    }

    public class SmallClass
    {
        public int N { get; set; }
    }
}
