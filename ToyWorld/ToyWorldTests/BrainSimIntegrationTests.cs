using GoodAI.TypeMapping;
using System;

namespace ToyWorldTests
{
    public class BrainSimIntegrationTests : IDisposable
    {
        public BrainSimIntegrationTests()
        {
            TypeMap.InitializeConfiguration<TestContainerConfiguration>();
        }

        public void Dispose()
        {
            TypeMap.Destroy();
        }
    }
}
