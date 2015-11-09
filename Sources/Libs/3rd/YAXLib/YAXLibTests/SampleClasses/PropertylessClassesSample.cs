using System;
using System.Reflection;

namespace YAXLibTests.SampleClasses
{
    public class PropertylessClassesSample
    {
        public DBNull ValuedDbNull { get; set; }
        public DBNull NullDbNull { get; set; }
        public object ObjValuedDbNull { get; set; }
        public object ObjNullDbNull { get; set; }

        public Random ValuedRandom { get; set; }
        public Random NullRandom { get; set; }
        public object ObjValuedRandom { get; set; }
        public object ObjNullRandom { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static PropertylessClassesSample GetSampleInstance()
        {
            return new PropertylessClassesSample
                       {
                           ValuedDbNull = DBNull.Value,
                           NullDbNull =  null,
                           ObjValuedDbNull = DBNull.Value,
                           ObjNullDbNull = null,
                           ValuedRandom = new Random(),
                           NullRandom = null,
                           ObjValuedRandom = new Random(),
                           ObjNullRandom = null
                       };
        }
    }
}
