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
        public void Convert9To10()
        {
            string xml = BasicNodesTests.Properties.Resources.Convert9To10XML;
            string expected = BasicNodesTests.Properties.Resources.Convert9To10XMLResult;
            string result = GoodAI.Modules.Versioning.MyConversion.Convert9To10(xml);

            Assert.Equal(expected, result);
        }
    }
}
