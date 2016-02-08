using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;
using Xunit;

namespace CoreTests
{
    public abstract class ContainerConfigurationTestBase : CoreTestBase
    {
        protected static void CheckResolveItem<T>() where T : class
        {
            var item = TypeMap.GetInstance<T>();
            Assert.NotNull(item);
        }

        [Fact]
        public void IsValid()
        {
            TypeMap.Verify();
        }
    }
}
