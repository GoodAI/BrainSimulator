using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Common;
using GoodAI.Modules.Motor;
using GoodAI.Modules.Testing;
using KellermanSoftware.CompareNetObjects;
using Xunit;
using Xunit.Abstractions;
using YAXLib;

namespace CoreTests
{
    public class SerializationTests : CoreTestBase
    {
        private ITestOutputHelper m_output;

        private class TestNode : MyWorkingNode
        {
            [MyBrowsable, Category("Communication"), YAXSerializableField(DefaultValue = "default")]
            public string TestProperty
            {
                get { return m_test; }
                set { m_test = value; }
            }

            private string m_test = "default";

            public override void UpdateMemoryBlocks()
            {
            }

            public TestTask Task { get; protected set; }

            public class TestTask : MyTask<TestNode>
            {
                public override void Init(int nGPU)
                {
                }

                public override void Execute()
                {
                }
            }
        }

        public SerializationTests(ITestOutputHelper output)
        {
            m_output = output;
        }

        [Fact]
        public void SerializesAndDeserializesCorrectly()
        {
            // I.e. deserialized(serialized(PROJECT)) should equal PROJECT
            string tmpPath = Path.GetTempPath();

            MyConfiguration.SetupModuleSearchPath();

            MyConfiguration.LoadModules();

            MyConfiguration.KnownNodes.Add(typeof(TestNode), new MyNodeConfig());

            var project = new MyProject();
            project.Network = project.CreateNode<MyNetwork>();
            project.Network.Name = "Network";
            project.CreateWorld(typeof (MyTestingWorld));
            project.Name = "test";
            var node = project.CreateNode<TestNode>();
            project.Network.AddChild(node);
            project.Restore();

            string serialized = project.Serialize(tmpPath);
            MyProject deserializedProject = MyProject.Deserialize(serialized, tmpPath);

            // MaxDifferences = 20 - A magic number. It shows more than one difference in the log.
            // There should eventually be zero differences, so this number can be arbitrary. Adjust as needed.
            // Observers are ignored - there are none, and the (de)serialization mechanism works with them in a special way.
            var compareLogic =
                new CompareLogic(new ComparisonConfig
                {
                    MaxDifferences = 20,
                    MembersToIgnore = new List<string> {"Observers"}
                });
            ComparisonResult result = compareLogic.Compare(project, deserializedProject);

            m_output.WriteLine(result.DifferencesString);

            Assert.True(result.AreEqual);
        }

        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
        private sealed class PropertyAddedTest
        {
            [YAXSerializableField]
            public string ExistingProperty { get; set; }
            [YAXSerializableField(DefaultValue = "bar")]
            public string NewProperty { get; set; }
        }

        [Fact]
        public void DeserializesNewPropertyWithDefaultValue()
        {
            var serializer = MyProject.GetSerializer<PropertyAddedTest>();

            var content = File.ReadAllText(@"Data\property_added_test.txt");
            var deserialized = serializer.Deserialize(content) as PropertyAddedTest;

            Assert.NotNull(deserialized);
            Assert.Equal("bar", deserialized.NewProperty);
        }

        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
        class PropertyMissingTest
        {
            [YAXSerializableField]
            public string ExistingProperty { get; set; }
        }

        [Fact]
        public void IgnoresMissingProperty()
        {
            var serializer = MyProject.GetSerializer<PropertyMissingTest>();

            var content = File.ReadAllText(@"Data\property_missing_test.txt");
            var deserialized = serializer.Deserialize(content) as PropertyMissingTest;

            Assert.NotNull(deserialized);
        }

        [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AttributedFieldsOnly)]
        class DictionaryTest
        {
            [YAXSerializableField, YAXDictionary(EachPairName = "Item")]
            public Dictionary<string, int> Dict { get; set; }

            public DictionaryTest()
            {
                Dict = new Dictionary<string, int>();
            }
        }

        [Fact]
        public void SerializesDictionary()
        {
            var serializer = MyProject.GetSerializer<DictionaryTest>();

            var dictTest = new DictionaryTest();
            dictTest.Dict.Add("foo", 1);
            dictTest.Dict.Add("bar", 2);

            var content = serializer.Serialize(dictTest);
            
            //File.WriteAllText(@"Data\serializes_dict.txt", content);

            Assert.Equal(File.ReadAllText(@"Data\serializes_dict.txt"), content);
        }
    }
}
