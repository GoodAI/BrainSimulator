using BrainSimulator.Utils;
using NDesk.Options;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using YAXLib;

namespace BrainSimulator.Configuration
{    
    public class MyConfiguration
    {
        public const string CORE_MODULE_NAME = "GoodAI.Platform.Core.dll";
        public const string CUSTOM_MODULES_NAME = "GoodAI.BasicNodes.dll";

        public static Dictionary<Type, MyNodeConfig> KnownNodes { get; private set; }
        public static Dictionary<Type, MyWorldConfig> KnownWorlds { get; private set; }

        public static List<MyModuleConfig> Modules { get; private set; }
        public static Dictionary<Assembly, MyModuleConfig> AssemblyLookup { get; private set; }

        public static List<string> ModulesSearchPath { get; private set; }
        public static string OpenOnStartupProjectName { get; private set; }

        public static string GlobalPTXFolder { get; private set; }

        static MyConfiguration()
        {
            GlobalPTXFolder = @".\ptx\";
            KnownNodes = new Dictionary<Type, MyNodeConfig>();
            KnownWorlds = new Dictionary<Type, MyWorldConfig>();
            Modules = new List<MyModuleConfig>();

            ModulesSearchPath = new List<string>();
            AssemblyLookup = new Dictionary<Assembly, MyModuleConfig>();
        }

        public static void SetupModuleSearchPath()
        {
            //SearchPath.Add(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData)); //add bs folder name

            foreach (string modulePath in Directory.GetDirectories(@"modules"))
            {
                ModulesSearchPath.Add(modulePath);
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
                    OpenOnStartupProjectName = Path.ChangeExtension(extraParams[0], ".brain");
                }
            }
            catch (OptionException e)
            {
                MyLog.ERROR.WriteLine(e.Message);
            }
        }

        public static void LoadModules()
        {
            MyLog.INFO.WriteLine("Loading system modules...");

            AddModuleFromAssembly(new FileInfo(CORE_MODULE_NAME), true);
            AddModuleFromAssembly(new FileInfo(CUSTOM_MODULES_NAME) );

            MyLog.INFO.WriteLine("Loading custom modules...");

            foreach (string modulePath in ModulesSearchPath)
            {
                FileInfo info = new FileInfo(modulePath);

                if ((info.Attributes & FileAttributes.Directory) > 0)
                {
                    info = new FileInfo(Path.Combine(info.FullName, info.Name + ".dll"));
                }

                if (info.Exists)
                {
                    AddModuleFromAssembly(info);
                }
                else
                {
                    MyLog.ERROR.WriteLine("Module assembly not found: " + info);
                }
            }
        }

        private static void AddModuleFromAssembly(FileInfo file, bool basicNode = false)
        {
            try
            {              
                MyModuleConfig moduleConfig = MyModuleConfig.LoadModuleConfig(file);                

                Modules.Add(moduleConfig);
                AssemblyLookup[moduleConfig.Assembly] = moduleConfig;

                if (moduleConfig.NodeConfigList != null)
                {
                    foreach (MyNodeConfig nc in moduleConfig.NodeConfigList)
                    {
                            nc.IsBasicNode = basicNode;
                            KnownNodes[nc.NodeType] = nc;
                        }
                        }

                if (moduleConfig.WorldConfigList != null)
                {
                    foreach (MyWorldConfig wc in moduleConfig.WorldConfigList)
                    {                       
                            wc.IsBasicNode = basicNode;
                            KnownWorlds[wc.NodeType] = wc;
                        }
                        }

                MyLog.INFO.WriteLine("Module loaded: " + file.Name 
                    + (moduleConfig.Conversion != null ? " (version=" + moduleConfig.GetXmlVersion() + ")" : " (no versioning)"));                
                    }
            catch (Exception e)
            {
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
}