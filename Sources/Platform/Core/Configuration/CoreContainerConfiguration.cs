using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;
using GoodAI.TypeMapping;
using SimpleInjector;

namespace GoodAI.Platform.Core.Configuration
{
    public class CoreContainerConfiguration : IContainerConfiguration
    {
        public void Configure(Container container)
        {
            container.Register<MyValidator>(Lifestyle.Singleton);
        }
    }
}
