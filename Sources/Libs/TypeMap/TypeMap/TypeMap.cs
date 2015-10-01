using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleInjector;

namespace GoodAI.TypeMapping
{
    public static class TypeMap
    {
        public static Container SimpleInjectorContainer { get; private set; }

        public static void Initialize()
        {
            SimpleInjectorContainer = new Container
            {
                Options =
                {
                    AllowOverridingRegistrations = true
                }
            };
        }

        public static void Destroy()
        {
            SimpleInjectorContainer.Dispose();
            SimpleInjectorContainer = null;
        }

        public static void InitializeConfiguration<TConfiguration>() where TConfiguration : IContainerConfiguration, new()
        {
            var configuration = new TConfiguration();
            configuration.Configure(SimpleInjectorContainer);
        }

        // TODO: add more of these delegations
        public static T GetInstance<T>() where T : class
        {
            return SimpleInjectorContainer.GetInstance<T>();
        }
    }
}
