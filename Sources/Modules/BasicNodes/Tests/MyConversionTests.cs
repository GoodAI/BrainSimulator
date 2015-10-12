using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace BasicNodesTests
{
    public class MyConversionTests
    {
        [Fact]
        public void Convert6To7()
        {
            string xml = BasicNodesTests.Properties.Resources.Convert6To7XML;
            string expected = BasicNodesTests.Properties.Resources.Convert6To7XMLResult;
            string result = GoodAI.Modules.Versioning.MyConversion.Convert6To7(xml);

            Assert.Equal(expected, result);
        }
    }
}
