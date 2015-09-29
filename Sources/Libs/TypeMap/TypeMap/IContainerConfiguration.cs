using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleInjector;

namespace GoodAI.TypeMapping
{
    public interface IContainerConfiguration
    {
        // TODO: add dependencies here?
        // Dependencies = a list of IContainerConfiguration types that this type needs for resolution.

        void Configure(Container container);
    }
}
