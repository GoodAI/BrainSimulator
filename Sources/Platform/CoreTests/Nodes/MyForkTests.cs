using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using Xunit;

namespace CoreTests.Nodes
{
    public class MyForkTests
    {
        [Fact]
        public void HandlesSimpleBranchConfig()
        {
            IList<int> branchSizes = MyFork.CalculateBranchSizes("2, 3", 5);

            Assert.Equal(2, branchSizes.Count);
            Assert.Equal(2, branchSizes[0]);
            Assert.Equal(3, branchSizes[1]);
        }

        [Fact]
        public void HandlesBranchConfigWithOneStar()
        {
            IList<int> branchSizes = MyFork.CalculateBranchSizes("10,*", 100);

            Assert.Equal(2, branchSizes.Count);
            Assert.Equal(10, branchSizes[0]);
            Assert.Equal(90, branchSizes[1]);
        }

        [Fact]
        public void HandlesMultipleStarsAndDivisionReminder()
        {
            IList<int> branchSizes = MyFork.CalculateBranchSizes("* , *, 5, *", 105);

            Assert.Equal(4, branchSizes.Count);
            Assert.Equal(33, branchSizes[0]);
            Assert.Equal(5, branchSizes[2]);
            Assert.Equal(34, branchSizes[3]);
        }

    }
}
