using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [ShowInDemoApplication]

    public class PathAndAliasAssignmentSample
    {
        [YAXAttributeFor("Title#value")]
        public string Title { get; set; }

        [YAXAttributeFor("Price#value")]
        public double Price { get; set; }

        [YAXAttributeFor("Publish#year")]
        public int PublishYear { get; set; }

        [YAXAttributeFor("Notes/Comments#value")]
        public string Comments { get; set; }

        [YAXAttributeFor("Author#name")]
        public string Author { get; set; }

        public override string ToString()
        {
            return GeneralToStringProvider.GeneralToString(this);
        }

        public static PathAndAliasAssignmentSample GetSampleInstance()
        {
            return new PathAndAliasAssignmentSample
            {
                Title = "Inside C#",
                Author = "Tom Archer & Andrew Whitechapel",
                PublishYear = 2002,
                Price = 30.5,
                Comments = "SomeComment",
            };
        }
    }
}
