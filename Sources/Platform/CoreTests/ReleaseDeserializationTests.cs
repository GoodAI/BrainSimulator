using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Dashboard;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;
using Xunit;

namespace CoreTests
{
    public class ReleaseDeserializationTests : CoreTestBase
    {
        [Fact]
        public void ReleaseDeserializationTest()
        {
            const string brainPath = @"Data\release-deserialization-test.brain";

            using (var runner = new MyProjectRunner())
            {
                runner.OpenProject(Path.GetFullPath(brainPath));

                // Must not fail.
                runner.RunAndPause(1);

                MyProject project = runner.Project;

                CheckDashboard(project);
            }
        }

        // Check the dashboard for expected properties.
        private static void CheckDashboard(MyProject project)
        {
            Dashboard dashboard = project.Dashboard;

            var property1 = CheckDashboardProperty(project, dashboard, 338, "PerformTask", "SimilarityOperator");
            
            var property2 = CheckDashboardProperty(project, dashboard, 331, "PerformTask", "SimilarityOperator");

            CheckDashboardProperty(project, dashboard, 451, "ShareWeightsTask", "SourceNodeName");

            MyWorkingNode nodeGroup = project.Network.GetChildNodeById(359) as MyWorkingNode;
            var property4 = dashboard.Get(nodeGroup, "InputBranches");
            Assert.NotNull(property4);

            // Check the grouped dashboard for expected properties.
            GroupDashboard groupedDashboard = project.GroupedDashboard;

            var group = groupedDashboard.Get("f6af17f3-82b0-42b6-89b0-a4eaf6432316");

            Assert.NotNull(@group);

            Assert.True(@group.GroupedProperties.Contains(property1));
            Assert.True(@group.GroupedProperties.Contains(property2));
        }

        private static DashboardNodePropertyBase CheckDashboardProperty(MyProject project, Dashboard dashboard, int nodeId,
            string taskName, string propertyName)
        {
            var node = project.Network.GetChildNodeById(nodeId) as MyWorkingNode;
            Assert.NotNull(node);

            MyTask task = node.GetTaskByPropertyName(taskName);
            Assert.NotNull(task);

            DashboardNodePropertyBase property = dashboard.Get(task, propertyName);
            Assert.NotNull(property);

            return property;
        }
    }
}
