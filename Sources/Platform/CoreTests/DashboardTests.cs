using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
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
    public class DashboardTests
    {
        private readonly ITestOutputHelper m_output;

        public class Node : MyWorkingNode
        {
            public override void UpdateMemoryBlocks()
            {
            }
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
            
            var property = new DashboardNodeProperty
            {
                Node = node,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };

            var group = new DashboardPropertyGroup();

            group.Add(property);

            Assert.Equal(group.PropertyName, property.Group.PropertyName);
            Assert.True(group.GroupedProperties.Contains(property));
        }

        [Fact]
        public void RemovesPropertyFromGroup()
        {
            var node = new Node();
            
            var property = new DashboardNodeProperty
            {
                Node = node,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };

            var group = new DashboardPropertyGroup();

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
            
            var property = new DashboardNodeProperty
            {
                Node = node,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };

            var proxy = property.Proxy;
            proxy.Value = testName;

            Assert.Equal(testName, node.Name);
        }

        [Fact]
        public void GroupProxyChangesValues()
        {
            const string testName = "TestName";

            var node = new Node();
            var node2 = new Node();
            
            var property = new DashboardNodeProperty
            {
                Node = node,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };

            var property2 = new DashboardNodeProperty
            {
                Node = node2,
                PropertyInfo = node2.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };

            var group = new DashboardPropertyGroup();

            group.Add(property);
            group.Add(property2);

            group.Proxy.Value = testName;

            Assert.Equal(testName, node.Name);
            Assert.Equal(testName, node2.Name);
        }

        /// <summary>
        /// The proxy property must return the same instance for the dashboard manipulation layer to work correctly.
        /// </summary>
        [Fact]
        public void ProxyIsTransient()
        {
            var node = new Node();
            var task = new Task();

            var property = new DashboardNodeProperty()
            {
                Node = node,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };
            Assert.Equal(property.Proxy, property.Proxy);

            var property2 = new DashboardTaskProperty()
            {
                Node = node,
                Task = task,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            };
            Assert.Equal(property2.Proxy, property2.Proxy);

            var property3 = new DashboardPropertyGroup();
            Assert.Equal(property3.Proxy, property3.Proxy);
        }

        [Fact]
        public void BothDashboardsSerialize()
        {
            var project = new MyProject();
            project.CreateWorld(typeof(MyTestingWorld));
            project.Network = new MyNetwork();
            var node = project.CreateNode<Node>();
            node.Name = "Foo";
            project.Network.AddChild(node);

            var dashboard = new Dashboard();
            dashboard.Properties.Add(new DashboardNodeProperty
            {
                Node = node,
                PropertyInfo = node.GetType().GetProperty("Name", BindingFlags.Public | BindingFlags.Instance)
            });

            var groupDashboard = new GroupDashboard();
            groupDashboard.Properties.Add(new DashboardPropertyGroup
            {
                PropertyName = "Group 1"
            });
            groupDashboard.Properties[0].Add(dashboard.Properties[0]);

            YAXSerializer serializer = MyProject.GetSerializer<Dashboard>();
            YAXSerializer groupSerializer = MyProject.GetSerializer<GroupDashboard>();
            string serializedDashboard = serializer.Serialize(dashboard);
            string serializedGroupDashboard = groupSerializer.Serialize(groupDashboard);

            Dashboard deserializedDashboard = (Dashboard) serializer.Deserialize(serializedDashboard);
            deserializedDashboard.RestoreFromIds(project);
            project.Dashboard = deserializedDashboard;

            GroupDashboard deserializedGroupDashboard = (GroupDashboard) groupSerializer.Deserialize(serializedGroupDashboard);
            deserializedGroupDashboard.RestoreFromIds(project);
            project.GroupedDashboard = deserializedGroupDashboard;

            var compareLogic = new CompareLogic(new ComparisonConfig
            {
                MaxDifferences = 20,
                MembersToIgnore = new List<string> { "Proxy", "GenericProxy" }
            });

            ComparisonResult result = compareLogic.Compare(dashboard, deserializedDashboard);
            m_output.WriteLine(result.DifferencesString);

            Assert.True(result.AreEqual);

            result = compareLogic.Compare(groupDashboard, deserializedGroupDashboard);
            m_output.WriteLine(result.DifferencesString);

            Assert.True(result.AreEqual);
        }
    }
}
