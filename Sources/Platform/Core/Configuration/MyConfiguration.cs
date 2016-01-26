using GoodAI.Core.Utils;
using NDesk.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using YAXLib;

namespace GoodAI.Core.Configuration
{    
    public class MyConfiguration
    {
        public const string CORE_MODULE_NAME = "GoodAI.Platform.Core.dll";
        public const string BASIC_NODES_MODULE_NAME = "GoodAI.BasicNodes.dll";
        public const string MODULES_PATH = @"modules";

        public static Dictionary<Type, MyNodeConfig> KnownNodes { get; private set; }
        public static Dictionary<Type, MyWorldConfig> KnownWorlds { get; private set; }

        public static Dictionary<string, MyCategoryConfig> KnownCategories { get; private set; }

        public static List<MyModuleConfig> Modules { get; private set; }
        public static Dictionary<string, MyModuleConfig> AssemblyLookup { get; private set; }

        public static List<string> ModulesSearchPath { get; private set; }
        public static string OpenOnStartupProjectName { get; private set; }

        public static string GlobalPTXFolder { get; private set; }        

        static MyConfiguration()
        {
            GlobalPTXFolder = MyResources.GetEntryAssemblyPath() + @"\modules\GoodAI.BasicNodes\ptx\";
            KnownNodes = new Dictionary<Type, MyNodeConfig>();
            KnownWorlds = new Dictionary<Type, MyWorldConfig>();
            KnownCategories = new Dictionary<string, MyCategoryConfig>();
            Modules = new List<MyModuleConfig>();

            ModulesSearchPath = new List<string>();
            AssemblyLookup = new Dictionary<string, MyModuleConfig>();
        }

        public static void SetupModuleSearchPath()
        {
            var modulesPath = Path.Combine(MyResources.GetEntryAssemblyPath(), MODULES_PATH);

            if (Directory.Exists(modulesPath))
            {
                foreach (string modulePath in Directory.GetDirectories(modulesPath))
                {
                    ModulesSearchPath.Add(modulePath);
                }
            }
        }

        public static void ProcessCommandParams()
        {            
            // using http://www.ndesk.org/Options for parsing options
            var options = new OptionSet() {                
                { "m|module=", "add a given module",
                    v => ModulesSearchPath.Add(v) },
//                { "h|?|help",  "shows this message and exit",
//                      v => show_help = v != null },
            };

            List<string> extraParams;
            try
            {
                extraParams = options.Parse(Environment.GetCommandLineArgs().Skip(1));
                if (extraParams.Count > 0)
                {
                    string brainFile = extraParams[0];
                    string extension = Path.GetExtension(brainFile);

                    if (extension != ".brain" && extension != ".brainz")
                        brainFile = Path.ChangeExtension(brainFile, ".brain");

                    OpenOnStartupProjectName = brainFile;
                }
            }
            catch (OptionException e)
            {
                MyLog.ERROR.WriteLine(e.Message);
            }
        }

        public static List<FileInfo> ListModules()
        {
            var moduleList = new List<FileInfo>();

            foreach (string modulePath in ModulesSearchPath)
            {
                var fileInfo = new FileInfo(modulePath);
                if ((fileInfo.Attributes & FileAttributes.Directory) > 0)
                    fileInfo = new FileInfo(Path.Combine(fileInfo.FullName, fileInfo.Name + ".dll"));

                if (!fileInfo.Exists)
                    MyLog.WARNING.WriteLine("Module assembly not found: " + fileInfo);

                moduleList.Add(fileInfo);
            }

            return moduleList;
        }

        public static void LoadModules()
        {
            MyLog.INFO.WriteLine("Loading system modules...");
            AddModuleFromAssembly(
                new FileInfo(Path.Combine(MyResources.GetEntryAssemblyPath(), CORE_MODULE_NAME)), basicNode: true);

            if (ModulesSearchPath.Count == 0)
                throw new InvalidOperationException("ModulesSearchPath must not be empty.");

            MyLog.INFO.WriteLine("Loading custom modules...");
            ListModules().ForEach(moduleFileInfo => AddModuleFromAssembly(moduleFileInfo));
        }

        private static void AddModuleFromAssembly(FileInfo file, bool basicNode = false)
        {
            try
            {
                MyModuleConfig moduleConfig = MyModuleConfig.LoadModuleConfig(file);

                Modules.Add(moduleConfig);
                AssemblyLookup[moduleConfig.Assembly.FullName] = moduleConfig;

                if (moduleConfig.NodeConfigList != null)
                {
                    foreach (MyNodeConfig nc in moduleConfig.NodeConfigList)
                    {
                        nc.IsBasicNode = basicNode;

                        if (nc.NodeType != null)
                        {
                            KnownNodes[nc.NodeType] = nc;
                        }
                    }
                }

                if (moduleConfig.WorldConfigList != null)
                {
                    foreach (MyWorldConfig wc in moduleConfig.WorldConfigList)
                    {
                        wc.IsBasicNode = basicNode;

                        if (wc.NodeType != null)
                        {
                            KnownWorlds[wc.NodeType] = wc;
                        }
                    }
                }

                if (moduleConfig.CategoryList != null)
                {
                    foreach (MyCategoryConfig categoryConfig in moduleConfig.CategoryList)
                    {
                        KnownCategories[categoryConfig.Name] = categoryConfig;
                    }
                }

                MyLog.INFO.WriteLine("Module loaded: " + file.Name
                                     +
                                     (moduleConfig.Conversion != null
                                         ? " (version=" + moduleConfig.GetXmlVersion() + ")"
                                         : " (no versioning)"));
            }
            catch (Exception e)
            {
                if (basicNode)
                {
                    throw new MyModuleLoadingException("Core module loading failed (" + e.Message + ")", e);
                }

                // We don't report the nodes.xml missing - the dll could still contain UI extensions.
                if (!(e is FileNotFoundException))
                    MyLog.ERROR.WriteLine("Module loading failed: " + e.Message);
            }
        }
    }

    [YAXSerializableType(FieldsToSerialize = YAXSerializationFields.AllFields),
    YAXSerializeAs("NetworkState")]
    public class MyNetworkState
    {
        [YAXSerializableField, YAXAttributeForClass, YAXSerializeAs("ForProject")]
        public string ProjectName { get; set; }

        [YAXSerializableField, YAXAttributeFor("MemoryBlocks"), YAXSerializeAs("Path")]
        public string MemoryBlocksLocation { get; set; }

        [YAXSerializableField, YAXAttributeFor("Simulation"), YAXSerializeAs("Step")]
        public uint SimulationStep { get; set; }
    }

    internal class MyModuleLoadingException : Exception 
    {
        public MyModuleLoadingException(string message, Exception innerException) : base(message, innerException) { }
    }
}