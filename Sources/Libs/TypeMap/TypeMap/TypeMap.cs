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
        private static Container SimpleInjectorContainer { get; set; }

        private static void Initialize()
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
            if (SimpleInjectorContainer == null)
                return;

            SimpleInjectorContainer.Dispose();
            SimpleInjectorContainer = null;
        }

        public static void InitializeConfiguration<TConfiguration>() where TConfiguration : IContainerConfiguration, new()
        {
            if (SimpleInjectorContainer == null)
                Initialize();

            var configuration = new TConfiguration();
            configuration.Configure(SimpleInjectorContainer);
        }

        // TODO: add more of these delegations
        public static T GetInstance<T>() where T : class
        {
            return SimpleInjectorContainer.GetInstance<T>();
        }

        public static void Verify()
        {
            SimpleInjectorContainer.Verify();
        }
    }
}
