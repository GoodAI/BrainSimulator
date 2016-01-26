using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using YAXLib;

namespace GoodAI.Core.Configuration
{
    [YAXSerializeAs("Configuration"), YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields)]
    public class MyModuleConfig
    {
        public const string MODULE_CONFIG_FILE = "nodes.xml";
        private const string CONVERSION_TYPE_NAME = "Versioning.MyConversion";

        [YAXSerializeAs("KnownNodes"), YAXErrorIfMissed(YAXExceptionTypes.Warning)]
        public List<MyNodeConfig> NodeConfigList = null;

        [YAXSerializeAs("KnownWorlds"), YAXErrorIfMissed(YAXExceptionTypes.Warning)]
        public List<MyWorldConfig> WorldConfigList = null;

        [YAXSerializeAs("Categories"), YAXErrorIfMissed(YAXExceptionTypes.Warning)]
        public List<MyCategoryConfig> CategoryList = null;

        [YAXAttributeForClass, YAXSerializableField(DefaultValue = "")]
        public string RootNamespace
        {
            get
            {
                return (String.IsNullOrEmpty(rootNamespace) && (this.Assembly != null)) ?
                    Assembly.GetName().Name  // (we could cache this if there's a performance problem)
                    :   
                    rootNamespace;
            }

            private set
            {
                rootNamespace = (value != null) ? value : String.Empty;
            }
        }
        [YAXDontSerialize]
        private string rootNamespace = String.Empty;

        [YAXDontSerialize]
        public Assembly Assembly { get; private set; }

        [YAXDontSerialize]
        public FileInfo File { get; private set; }

        [YAXDontSerialize]
        public MyBaseConversion Conversion { get; private set; }

        public MyModuleConfig()
        {
        }

        public static MyModuleConfig LoadModuleConfig(FileInfo assemblyFile)
        {
            Assembly assembly = Assembly.LoadFrom(assemblyFile.FullName);
            string xml = MyResources.GetTextFromAssembly(assembly, MODULE_CONFIG_FILE);

            if (string.IsNullOrEmpty(xml))
            {
                throw new FileNotFoundException("Module config not found (resource \"nodes.xml\" missing for module " + assemblyFile.Name + ").");
            }

            YAXSerializer serializer = new YAXSerializer(typeof(MyModuleConfig),
                   YAXExceptionHandlingPolicies.ThrowErrorsOnly, YAXExceptionTypes.Warning);            

            MyModuleConfig moduleConfig = (MyModuleConfig)serializer.Deserialize(xml);            

            if (moduleConfig == null)
            {
                throw new YAXException("Module config parsing failed: " + serializer.ParsingErrors);
            }

            moduleConfig.Assembly = assembly;
            moduleConfig.File = assemblyFile;

            moduleConfig.LoadConversionClass();

            moduleConfig.FinalizeNodeConfigs<MyNodeConfig>(moduleConfig.NodeConfigList);
            moduleConfig.FinalizeNodeConfigs<MyWorldConfig>(moduleConfig.WorldConfigList);
            moduleConfig.FinalizeCategoriesConfig();

            return moduleConfig;
        }

        private void FinalizeNodeConfigs<T>(List<T> nodeList) where T : MyNodeConfig
        {
            if (nodeList != null)
            {
                foreach (T nc in nodeList)
                {
                    try
                    {
                        nc.FinalizeConfig(Assembly);
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Node type loading failed: " + e.Message);
                    }
                }
            }
        }

        private void FinalizeCategoriesConfig()
        {
            if (CategoryList == null)
                return;

            foreach (MyCategoryConfig categoryConfig in CategoryList)
            {
                categoryConfig.InitIcons(Assembly);
            }
        }

        private void LoadConversionClass()
        {
            string typeName = RootNamespace + "." + CONVERSION_TYPE_NAME;

            try
            {
                Type conversionType = Assembly.GetType(typeName);

                if (conversionType != null)
                {
                    Conversion = (MyBaseConversion)Activator.CreateInstance(conversionType);
                    Conversion.Module = this;
                }
            }
            catch
            {
                Conversion = null;
            }

            if (Conversion == null)
            {
                MyLog.WARNING.WriteLine("Can't load version (looking for type {0}).", typeName);
            }
        }

        public int GetXmlVersion()
        {
            return Conversion != null ? Conversion.CurrentVersion : 1;
        }
    }
}
