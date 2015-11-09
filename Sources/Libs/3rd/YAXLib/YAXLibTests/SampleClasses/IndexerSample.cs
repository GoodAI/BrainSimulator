namespace YAXLibTests.SampleClasses
{
    public class IndexerSample
    {
        public int this[int i]
        {
            get { return i * 2; }
            set { SomeInt = i * value; }
        }

        public int SomeInt { get; set; }

        public string SomeString { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static IndexerSample GetSampleInstance()
        {
            return new IndexerSample
                   {
                       SomeInt = 1234,
                       SomeString = "Something"
                   };
        }
    }
}
