using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Nodes;
using GoodAI.TypeMapping;
using SimpleInjector;

namespace GoodAI.Platform.Core.Configuration
{
    public class CoreContainerConfiguration : IContainerConfiguration
    {
        public void Configure(Container container)
        {
            container.Register<MyValidator>(Lifestyle.Singleton);
            container.Register<IMyExecutionPlanner, MyDefaultExecutionPlanner>(Lifestyle.Singleton);
            container.Register<MySimulation, MyLocalSimulation>(Lifestyle.Singleton);

            container.Register<IModelChanges, ModelChanges>(Lifestyle.Transient);

            container.Register<IMemoryBlockMetadata, MemoryBlockMetadata>(Lifestyle.Transient);
        }
    }
}
