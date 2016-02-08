using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;

namespace CoreTests
{
    public abstract class CoreTestBase : IDisposable
    {
        protected CoreTestBase()
        {
            TypeMap.InitializeConfiguration<CoreContainerConfiguration>();
        }

        public void Dispose()
        {
            TypeMap.Destroy();
        }
    }
}
