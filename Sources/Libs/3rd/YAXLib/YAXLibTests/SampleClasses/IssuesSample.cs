using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;
using YAXLib;

namespace YAXLibTests.SampleClasses
{
    [YAXSerializeAs("issues")]
    public class IssuesSample
    {
        [YAXCollection(YAXCollectionSerializationTypes.RecursiveWithNoContainingElement, EachElementName="issue")]
        public List<Issue> Issues { get; set; }

        [YAXSerializeAs("type")]
        [YAXAttributeForClass]
        public string Type { get; set; }

        [YAXSerializeAs("count")]
        [YAXAttributeForClass]
        public int Count { get; set; }

        public static IssuesSample GetSampleInstance()
        {
            List<Issue> issues = new List<Issue>();

            Issue issue1 = new Issue()
            {
                IssueID = 425,
                ProjectName = "Tech Pubs Box",
                ProjectId = 141,
                TrackerName = "Bug",
                TrackerId = 1,
                Subject = "::project::Platform test",
                Description = "",
                StartDate = new DateTime(2010, 5, 20),
                DueDate = new DateTime(2010, 5, 20),
                CreatedOn = new DateTime(2010, 5, 20, 14, 19, 59, 700),
                UpdatedOn = new DateTime(2010, 5, 20, 14, 20, 37, 700),
                CustomFields = new List<CustomField>() 
                { 
                    new CustomField("Provided Steps to Reproduce", 69, "0"),
                    new CustomField("Steps to Reproduce", 71, ""),
                    new CustomField("Browser", 1, ""),
                    new CustomField("Platform", 77, "n/a"),
                    new CustomField("Workaround", 49, ""),
                    new CustomField("Customer", 47, ""),
                }
            };

            issues.Add(issue1);
            return new IssuesSample()
            {   Type = "array",
                Count = 22,
                Issues = issues
            };
        }
    }

    public class Issue
    {
        [YAXSerializeAs("id")]
        public int IssueID { get; set; }

        [YAXSerializeAs("name")]
        [YAXAttributeFor("project")]
        public string ProjectName { get; set; }

        [YAXSerializeAs("id")]
        [YAXAttributeFor("project")]
        public int ProjectId { get; set; }

        [YAXSerializeAs("name")]
        [YAXAttributeFor("tracker")]
        public string TrackerName { get; set; }

        [YAXSerializeAs("id")]
        [YAXAttributeFor("tracker")]
        public int TrackerId { get; set; }

             // do the same for status, priority, author

        [YAXSerializeAs("subject")]
        public string Subject { get; set; }

        [YAXSerializeAs("description")]
        public string Description { get; set; }

        [YAXSerializeAs("start_date")]
        [YAXFormat("yyyy-MM-dd")]
        public DateTime StartDate { get; set; }

        [YAXSerializeAs("due_date")]
        [YAXFormat("yyyy-MM-dd")]
        public DateTime DueDate { get; set; }

        // and so on

        [YAXCollection(YAXCollectionSerializationTypes.Recursive, EachElementName="custom_field")]
        [YAXSerializeAs("custom_fields")]
        public List<CustomField> CustomFields { get; set; }

        [YAXSerializeAs("created_on")]
        [YAXFormat("R")]
        public DateTime CreatedOn { get; set; }

        [YAXSerializeAs("updated_on")]
        [YAXFormat("R")]
        public DateTime UpdatedOn { get; set; }
    }

    public class CustomField
    {
        [YAXSerializeAs("name")]
        [YAXAttributeForClass]
        public string Name { get; set; }

        [YAXSerializeAs("id")]
        [YAXAttributeForClass]
        public int ID { get; set; }

        [YAXSerializeAs("value")]
        [YAXAttributeForClass]
        public string Value { get; set; }

        public CustomField(string name, int id, string value)
        {
            this.Name = name;
            this.ID = id;
            this.Value = value;
        }
    }
}
