using GoodAI.Core.Configuration;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using WeifenLuo.WinFormsUI.Docking;

namespace GoodAI.BrainSimulator.Forms
{
    [AttributeUsage(AttributeTargets.Class)]
    public class BrainSimUIExtensionAttribute : Attribute { }

    public static class UIPlugins
    {
        internal static IEnumerable<DockContent> GetBrainSimUIExtensions(MainForm mainForm)
        {
            List<DockContent> result = new List<DockContent>();

            foreach (Type test in GetBrainSimUIExtensionTypes())
            {
                ConstructorInfo defaultCtor = test.GetConstructor(Type.EmptyTypes);
                ConstructorInfo mainformCtor = test.GetConstructor(new Type[] { typeof(MainForm) });
                if (mainformCtor != null)
                    result.Add((DockContent)mainformCtor.Invoke(new object[] { mainForm }));
                else if (defaultCtor != null)
                    result.Add((DockContent)defaultCtor.Invoke(new object[] { }));
            }

            return result;
        }

        private static IEnumerable<Type> GetBrainSimUIExtensionTypes()
        {
            var ret = new List<Type>();

            foreach (FileInfo assemblyFile in MyConfiguration.ListModules())
            {
                try
                {
                    Assembly assembly = Assembly.LoadFrom(assemblyFile.FullName);
                    string xml;
                    if (MyResources.TryGetTextFromAssembly(assembly, MyModuleConfig.MODULE_CONFIG_FILE, out xml))
                        continue;

                    MyLog.INFO.WriteLine("UI module loaded: " + assemblyFile.Name);
                    ret.AddRange(assembly.GetTypes().Where(IsUIExtension));
                }
                catch (Exception ex)
                {
                    MyLog.DEBUG.WriteLine("Error when looking for UI modules: " + ex.Message);
                }
            }

            return ret;
        }

        private static bool IsUIExtension(Type type)
        {
            return Attribute.GetCustomAttribute(type, typeof(BrainSimUIExtensionAttribute)) != null &&
                   type.IsSubclassOf(typeof(DockContent));
        }
    }
}

