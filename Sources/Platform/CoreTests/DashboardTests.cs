using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Resources;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Motor;
using GoodAI.Modules.Testing;
using KellermanSoftware.CompareNetObjects;
using Xunit;
using Xunit.Abstractions;
using YAXLib;

namespace CoreTests
{
    public class DashboardTests : CoreTestBase
    {
        private readonly ITestOutputHelper m_output;

        public class Node : MyWorkingNode
        {
            public override void UpdateMemoryBlocks()
            {
            }

            public Task Task { get; set; }

            [MyTaskGroup("TestGroup")]
            public Task Task2 { get; set; }

            [MyTaskGroup("TestGroup")]
            public Task Task3 { get; set; }
        }

        public class Task : MyTask<Node>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
            }
        }

        public DashboardTests(ITestOutputHelper output)
        {
            m_output = output;
        }

        [Fact]
        public void AddsPropertyToGroup()
        {
            var node = new Node();

            var property = GetDirectProperty(node);

            var group = new DashboardPropertyGroup("Foo");

            group.Add(property);

            Assert.Equal(group.PropertyName, property.Group.PropertyName);
            Assert.True(group.GroupedProperties.Contains(property));
        }

        [Fact]
        public void RemovesPropertyFromGroup()
        {
            var node = new Node();

            var property = GetDirectProperty(node);

            var group = new DashboardPropertyGroup("Foo");

            group.Add(property);
            group.Remove(property);

            Assert.Null(property.Group);
            Assert.Empty(group.GroupedProperties);
        }

        [Fact]
        public void ProxyChangesValue()
        {
            const string testName = "TestName";

            var node = new Node();

            var property = GetDirectProperty(node);

            var proxy = property.GenericProxy;
            proxy.Value = testName;

            Assert.Equal(testName, node.Name);
            Assert.Equal(testName, proxy.Value);
        }

        [Fact]
        public void GroupProxyChangesValues()
        {
            const string testName = "TestName";

            var node = new Node();
            var node2 = new Node();

            var property = GetDirectProperty(node);

            var property2 = GetDirectProperty(node2);

            var group = new DashboardPropertyGroup("Foo");

            group.Add(property);
            group.Add(property2);

            group.GenericProxy.Value = testName;

            Assert.Equal(testName, node.Name);
            Assert.Equal(testName, node2.Name);
            Assert.Equal(testName, group.GenericProxy.Value);
        }

        /// <summary>
        /// The proxy property must return the same instance for the dashboard manipulation layer to work correctly.
        /// </summary>
        [Fact]
        public void ProxyIsTransient()
        {
            var node = new Node();
            var task = new Task();

            var property = GetDirectProperty(node);
            Assert.Equal(property.GenericProxy, property.GenericProxy);

            var property2 = new DashboardTaskProperty(task, node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance));
            Assert.Equal(property2.GenericProxy, property2.GenericProxy);

            var property3 = new DashboardPropertyGroup("Foo");
            Assert.Equal(property3.GenericProxy, property3.GenericProxy);
        }

        [Fact]
        public void DescriptorHasCorrectType()
        {
            var node = new Node();

            var property = GetDirectProperty(node);

            var proxy = property.GenericProxy;
            var descriptor = new ProxyPropertyDescriptor(ref proxy, new Attribute[0]);

            Assert.Equal(descriptor.PropertyType, typeof (string));
        }

        [Fact]
        public void GroupDescriptorHasCorrectType()
        {
            var node = new Node();

            var property = GetDirectProperty(node);

            var group = new DashboardPropertyGroup("Foo");
            group.Add(property);

            var proxy = group.GenericProxy;
            var descriptor = new ProxyPropertyDescriptor(ref proxy, new Attribute[0]);

            Assert.Equal(descriptor.PropertyType, typeof (string));
        }

        private static DashboardNodeProperty GetDirectProperty(Node node)
        {
            var property = new DashboardNodeProperty(node,
                node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance));
            return property;
        }

        [Fact]
        public void DashboardFiresUpdateEvents()
        {
            var node = new Node();

            var flag = new AutoResetEvent(false);

            var dashboard = new Dashboard();
            dashboard.PropertyChanged += (sender, args) => flag.Set();

            dashboard.Add(node, "Name");

            if (!flag.WaitOne(1000))
                Assert.True(false, "Event not fired");

            dashboard.Remove(node, "Name");

            if (!flag.WaitOne(1000))
                Assert.True(false, "Event not fired");
        }

        [Fact]
        public void GroupDashboardFiresUpdateEvents()
        {
            var flag = new AutoResetEvent(false);

            var groupDashboard = new GroupDashboard();
            groupDashboard.PropertyChanged += (sender, args) => flag.Set();

            groupDashboard.Add();

            if (!flag.WaitOne(1000))
                Assert.True(false, "Event not fired");

            groupDashboard.Remove(groupDashboard.Properties[0]);

            if (!flag.WaitOne(1000))
                Assert.True(false, "Event not fired");
        }

        [Fact]
        public void BothDashboardsSerialize()
        {
            var project = new MyProject();
            project.CreateWorld(typeof (MyTestingWorld));
            project.Network = new MyNetwork();
            var node = project.CreateNode<Node>();
            node.Name = "Foo";
            project.Network.AddChild(node);

            var dashboard = new Dashboard();

            foreach (PropertySetup propertySetup in GetPropertyExamples(project))
                dashboard.Add(propertySetup.Target, propertySetup.PropertyName);

            var groupDashboard = new GroupDashboard();
            groupDashboard.Properties.Add(new DashboardPropertyGroup("Group 1"));
            groupDashboard.Properties[0].Add(dashboard.Properties[0]);

            YAXSerializer serializer = MyProject.GetSerializer<Dashboard>();
            YAXSerializer groupSerializer = MyProject.GetSerializer<GroupDashboard>();
            string serializedDashboard = serializer.Serialize(dashboard);
            string serializedGroupDashboard = groupSerializer.Serialize(groupDashboard);

            Dashboard deserializedDashboard = (Dashboard) serializer.Deserialize(serializedDashboard);
            deserializedDashboard.RestoreFromIds(project);
            project.Dashboard = deserializedDashboard;

            GroupDashboard deserializedGroupDashboard =
                (GroupDashboard) groupSerializer.Deserialize(serializedGroupDashboard);
            deserializedGroupDashboard.RestoreFromIds(project);
            project.GroupedDashboard = deserializedGroupDashboard;

            var compareLogic = new CompareLogic(new ComparisonConfig
            {
                MaxDifferences = 20,
                MembersToIgnore = new List<string> {"Proxy", "GenericProxy"}
            });

            ComparisonResult result = compareLogic.Compare(dashboard, deserializedDashboard);
            m_output.WriteLine(result.DifferencesString);

            Assert.True(result.AreEqual);

            result = compareLogic.Compare(groupDashboard, deserializedGroupDashboard);
            m_output.WriteLine(result.DifferencesString);

            Assert.True(result.AreEqual);
        }

        private class PropertySetup
        {
            public object Target { get; set; }
            public string PropertyName { get; set; }

            public PropertySetup(object target, string propertyName)
            {
                Target = target;
                PropertyName = propertyName;
            }
        }

        private static IEnumerable<PropertySetup> GetPropertyExamples(MyProject project)
        {
            var node = project.GetNodeById(1) as Node;
            return new List<PropertySetup>
            {
                new PropertySetup(node, "Name"),
                new PropertySetup(node.Task, "Name"),
                new PropertySetup(node.TaskGroups["TestGroup"], "TestGroup")
            };
        }

        [Fact]
        public void ChecksNameUniqueness()
        {
            var dashboard = new GroupDashboard();
            dashboard.Add();
            dashboard.Add();

            var group1 = dashboard.Properties[0];
            var group2 = dashboard.Properties[1];

            group1.PropertyName = "a";
            group2.PropertyName = "b";

            Assert.True(dashboard.CanChangeName(group1, "a"));
            Assert.False(dashboard.CanChangeName(group1, "b"));
        }

        [Fact]
        public void LooksUpGroupByName()
        {
            var dashboard = new GroupDashboard();
            dashboard.Add();
            dashboard.Add();

            var group1 = dashboard.Properties[0];
            var group2 = dashboard.Properties[1];

            group1.PropertyName = "a";
            group2.PropertyName = "b";

            Assert.Equal(group1, dashboard.GetByName(group1.PropertyName));
            Assert.Equal(group2, dashboard.GetByName(group2.PropertyName));
        }

        [Fact]
        public void GeneratesUniqueNameForGroups()
        {
            // If multiple groups are found in the deserialized file with the same name, only one group keeps the name
            // and the others get a new, unique name.

            var project = new MyProject();

            project.Dashboard = new Dashboard();
            project.GroupedDashboard = new GroupDashboard();
            project.GroupedDashboard.Add();
            project.GroupedDashboard.Add();

            var group1 = project.GroupedDashboard.Properties[0];
            var group2 = project.GroupedDashboard.Properties[1];

            group1.PropertyName = group2.PropertyName = "abc";

            project.GroupedDashboard.RestoreFromIds(project);

            Assert.False(group1.PropertyName == group2.PropertyName);
        }
    }
}