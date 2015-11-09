using System;

namespace YAXLibTests
{
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct)]
    public class ShowInDemoApplicationAttribute : Attribute
    {
        public ShowInDemoApplicationAttribute()
        {
            SortKey = null;
            GetSampleInstanceMethodName = "GetSampleInstance";
        }

        public string SortKey { get; set; }

        public string GetSampleInstanceMethodName { get; set; }
    }
}
