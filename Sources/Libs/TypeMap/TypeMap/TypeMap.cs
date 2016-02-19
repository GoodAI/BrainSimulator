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
        public static Container SimpleInjectorContainer
        {
            get
            {
                if (m_simpleInjectorContainer == null)
                    Initialize();

                return m_simpleInjectorContainer;
            }
        }
        private static Container m_simpleInjectorContainer;

        private static void Initialize()
        {
            m_simpleInjectorContainer = new Container
            {
                Options =
                {
                    AllowOverridingRegistrations = true
                }
            };
        }

        public static void Destroy()
        {
            if (m_simpleInjectorContainer == null)
                return;

            m_simpleInjectorContainer.Dispose();
            m_simpleInjectorContainer = null;
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

        public static void Verify()
        {
            SimpleInjectorContainer.Verify();
        }
    }
}
