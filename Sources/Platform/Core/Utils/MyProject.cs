using GoodAI.Core.Configuration;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Execution;
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using GoodAI.Core.Dashboard;
using YAXLib;

namespace GoodAI.Core.Utils
{
    [YAXSerializeAs("Project"), YAXSerializableType(FieldsToSerialize=YAXSerializationFields.AttributedFieldsOnly)]
    public class MyProject : IDisposable
    {        
        [YAXSerializableField, YAXAttributeForClass]
        public string Name { get; set; }

        public void SetNameFromPath(string path)
        {
            Name = MakeNameFromPath(path);
        }

        public static string MakeNameFromPath(string path)
        {
            return Path.GetFileNameWithoutExtension(path);
        }

        public static string MakeDataFolderFromFileName(string path)
        {
            return Path.Combine(Path.GetDirectoryName(path), MyProject.MakeNameFromPath(path) + ".statedata");
        }

        [YAXSerializableField]
        [YAXSerializeAs("Observers")]
        public List<MyAbstractObserver> Observers { get; set; }

        public MySimulationHandler SimulationHandler { get; set; }

        public void Dispose()
        {
            DisconnectWorld();

            Network = null;
            World = null;
        }

        #region Node creation

        private int m_nodeCounter = 0;
        
        public int GenerateNodeId()
        {
            int id = m_nodeCounter;
            m_nodeCounter++;

            return id;
        }

        public MyNode GetNodeById(int nodeId)
        {
            if (World.Id == nodeId)
            {
                return World;
            }
            else
            {
                return Network.GetChildNodeById(nodeId);
            }
        }

        public TNode CreateNode<TNode>() where TNode : MyNode, new()
        {
            return (TNode)CreateNode(typeof(TNode));
        }        

        public MyNode CreateNode(Type nodeType)
        {
            var newNode = Activator.CreateInstance(nodeType) as MyNode;
            newNode.Owner = this;
            newNode.Init();

            return newNode;
        }

        #endregion

        public MyConnection Connect(MyNode fromNode, MyNode toNode, int fromIndex=0, int toIndex=0)
        {
            var connection = new MyConnection(fromNode, toNode, fromIndex, toIndex);
            connection.Connect();

            return connection;
        }

        #region Network & World properties code

        private MyNetwork m_network;
        [YAXSerializableField]
        public MyNetwork Network
        {
            get { return m_network; }
            set
            {
                if (m_network != null)
                {
                    m_network.Dispose();
                }
                m_network = value;

                if (m_network != null && m_network.Owner == null)
                {
                    m_network.Owner = this;
                }
            }
        }

        private MyWorld m_world;

        [YAXSerializableField]
        public MyWorld World
        {
            get { return m_world; }
            internal set
            {
                if (m_world != null)
                {
                    m_world.Dispose();
                }
                m_world = value;

                if (m_world != null && m_world.Owner == null)
                {
                    m_world.Owner = this;
                }
            }
        }

        public void CreateWorld(Type worldType)
        {
            DisconnectWorld();

            World = Activator.CreateInstance(worldType) as MyWorld;
            World.Owner = this;
            World.Init();

            World.Name = "World";

            ConnectWorld();
        }

        private void DisconnectWorld()
        {
            if (Network != null && World != null)
            {
                foreach (MyConnection c in Network.InputConnections)
                {
                    if (c != null)
                    {
                        c.Disconnect();
                    }
                }

                foreach (MyConnection c in World.InputConnections)
                {
                    if (c != null)
                    {
                        c.Disconnect();
                    }
                }
            }
        }

        private void ConnectWorld()
        {
            if (Network != null && World != null)
            {
                if (Network.InputBranches != World.OutputBranches)
                {
                    Network.InputBranches = World.OutputBranches;
                    Network.InitInputNodes();
                }

                for (int i = 0; i < World.OutputBranches; i++)
                {
                    Network.GroupInputNodes[i].Name = ShortenMemoryBlockName(World.GetOutput(i).Name);
                }

                if (Network.OutputBranches != World.InputBranches)
                {
                    Network.OutputBranches = World.InputBranches;
                    Network.InitOutputNodes();
                }

                for (int i = 0; i < World.InputBranches; i++)
                {
                    Network.GroupOutputNodes[i].Name = ShortenMemoryBlockName(MyNodeInfo.Get(World.GetType()).InputBlocks[i].Name);
                }

                for (int i = 0; i < World.OutputBranches; i++)
                {
                    MyConnection worldToNetwork = new MyConnection(World, Network, i, i);
                    worldToNetwork.Connect();
                }

                for (int i = 0; i < World.InputBranches; i++)
                {
                    MyConnection networkToWorld = new MyConnection(Network, World, i, i);
                    networkToWorld.Connect();
                }
            }
        }

        #endregion

        #region Dashboard

        [YAXSerializableField]
        public Dashboard.Dashboard Dashboard { get; set; }

        [YAXSerializableField]
        public Dashboard.GroupDashboard GroupedDashboard { get; set; }

        #endregion

        #region Project Options

        [YAXSerializableField]
        public bool LoadAllNodesData { get; set; }

        [YAXSerializableField]
        public bool SaveAllNodesData { get; set; }

        #endregion

        #region Serialization & Versioning

        [YAXSerializableField, YAXSerializeAs("UsedModules")]
        internal List<MyUsedModuleInfo> UsedModules;

        public bool ReadOnly { get; internal set; }

        public static YAXSerializer GetSerializer()
        {
            return GetSerializer<MyProject>();
        }

        public static YAXSerializer GetSerializer<T>()
        {
            return new YAXSerializer(typeof(T),
                YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Error, YAXSerializationOptions.SerializeNullObjects);
        }

        public string Serialize(string projectPath)
        {                        
            if (ReadOnly)            
            {
                throw new InvalidOperationException("Project was not loaded correctly. Save is not allowed.");                
            }
            
            Network.PrepareConnections();
            UsedModules = ScanForUsedModules();

            StringBuilder sb = new StringBuilder("<?xml version=\"1.0\" encoding=\"utf-8\"?>" + System.Environment.NewLine);
            MyPathSerializer.ReferencePath = projectPath;  // needed for conversion of absolute paths to relative ones
            sb.Append(MyProject.GetSerializer().Serialize(this));
            MyPathSerializer.ReferencePath = String.Empty;

            return sb.ToString();
        }

        private List<MyUsedModuleInfo> ScanForUsedModules()
        {            
            HashSet<MyModuleConfig> usedModules = new HashSet<MyModuleConfig>();
         
            MyNodeGroup.IteratorAction scanForModules = delegate(MyNode node)
            {
                if (MyConfiguration.AssemblyLookup.ContainsKey(node.GetType().Assembly.FullName))
                {
                    usedModules.Add(MyConfiguration.AssemblyLookup[node.GetType().Assembly.FullName]);
                }
                else
                {
                    MyLog.WARNING.WriteLine("Unregistered module used in the project: " + node.GetType().Assembly.FullName);
                }
            };

            Network.Iterate(true, scanForModules);
            scanForModules(Network);
            scanForModules(World);            

            List<MyUsedModuleInfo> result = new List<MyUsedModuleInfo>();

            foreach (MyModuleConfig module in usedModules)
            {
                result.Add(new MyUsedModuleInfo()
                {
                    Name = module.File.Name,
                    Version = module.GetXmlVersion()
                });
            }

            return result;
        }

        private static string CheckUsedModulesAndConvert(string xml)
        {
            Dictionary<string, MyModuleConfig> moduleLookup = new Dictionary<string, MyModuleConfig>();            

            foreach (MyModuleConfig module in MyConfiguration.Modules)
            {
                moduleLookup[module.File.Name] = module;
            }

            List<MyUsedModuleInfo> usedModules = MyUsedModuleInfo.DeserializeUsedModulesSection(xml);

            string convertedXml = xml;            

            foreach (MyUsedModuleInfo usedModule in usedModules)
            {
                if (moduleLookup.ContainsKey(usedModule.Name))
                {
                    MyModuleConfig moduleConfig = moduleLookup[usedModule.Name];

                    if (moduleConfig.Conversion != null)
                    {
                        convertedXml = moduleConfig.Conversion.ApplyConversionsIfNeeed(convertedXml, usedModule.Version);
                    }
                }
                else
                {
                    MyLog.ERROR.WriteLine("Referenced module not available: " + usedModule.Name);
                }
            }

            return convertedXml;
        }        

        /// <summary>
        /// Deserializes the project from a given string.
        /// </summary>
        /// <param name="xml">The input string for deserialization.</param>
        /// <param name="projectPath">Project path for correct lookup of items like state data.</param>
        /// <param name="restoreModelOnly">If set to true, only the model is deserialized, but not observers etc.</param>
        /// <returns>A deserialized project.</returns>
        public static MyProject Deserialize(string xml, string projectPath, bool restoreModelOnly = false)
        {            
            xml = MyBaseConversion.ConvertOldFileVersioning(xml);
            xml = MyBaseConversion.ConvertOldModuleNames(xml);

            xml = CheckUsedModulesAndConvert(xml);
            
            YAXSerializer serializer = MyProject.GetSerializer();
            MyPathSerializer.ReferencePath = projectPath;
            MyProject loadedProject = (MyProject)serializer.Deserialize(xml);
            MyPathSerializer.ReferencePath = String.Empty;

            DumpSerializerErrors(serializer);
            
            if (loadedProject == null)
            {
                throw new YAXException("Cannot deserialize project.");
            }

            loadedProject.World.FinalizeTasksDeserialization();

            loadedProject.World.UpdateAfterDeserialization();
            loadedProject.m_nodeCounter = loadedProject.Network.UpdateAfterDeserialization(0, loadedProject);

            if (loadedProject.World.Id > loadedProject.m_nodeCounter)
            {
                loadedProject.m_nodeCounter = loadedProject.World.Id;
            }

            loadedProject.m_nodeCounter++;

            loadedProject.ConnectWorld();            

            if (!restoreModelOnly)
                loadedProject.Restore();

            return loadedProject;
        }

        private static void DumpSerializerErrors(YAXSerializer serializer)
        {
            for (int i = 0; i < serializer.ParsingErrors.Count; i++)
            {
                var error = serializer.ParsingErrors[i];

                MyLogLevel logLevel;

                switch (error.Value)
                {
                    case YAXExceptionTypes.Error:
                        logLevel = MyLogLevel.ERROR;
                        break;
                    case YAXExceptionTypes.Ignore:
                        logLevel = MyLogLevel.INFO;
                        break;
                    case YAXExceptionTypes.Warning:
                        logLevel = MyLogLevel.WARNING;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                if (error.Key is YAXAttributeMissingException || error.Key is YAXElementValueMissingException)
                    logLevel = MyLogLevel.WARNING;

                if (error.Key is YAXElementMissingException)
                    logLevel = MyLogLevel.DEBUG;

                switch (logLevel)
                {
                    case MyLogLevel.ERROR:
                        MyLog.ERROR.WriteLine(error.Key.Message);
                        break;
                    case MyLogLevel.WARNING:
                        MyLog.WARNING.WriteLine(error.Key.Message);
                        break;
                    case MyLogLevel.DEBUG:
                        MyLog.DEBUG.WriteLine(error.Key.Message);
                        break;
                    case MyLogLevel.INFO:
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        /*
        public void ConvertPathsToRelative(string referencePath)
        {
            Uri referenceUri = new Uri(referencePath, UriKind.Absolute);

            MyNodeGroup.IteratorAction convertAction = delegate(MyNode target)
            {
                if (target is MyWorkingNode)
                {
                    MyWorkingNode workingNode = target as MyWorkingNode;

                    if (!string.IsNullOrEmpty(workingNode.DataFolder))
                    {
                        Uri uri = new Uri(workingNode.DataFolder, UriKind.RelativeOrAbsolute);

                        if (uri.IsAbsoluteUri)
                        {
                            Uri relativeUri = referenceUri.MakeRelativeUri(uri);
                            workingNode.DataFolder = relativeUri.OriginalString.Replace('/', '\\');
                        }
                    }
                }
            };

            convertAction(World);
            Network.Iterate(true, convertAction);
        }

        public void ConvertPathsToAbsolute(string referencePath)
        {
            MyNodeGroup.IteratorAction convertAction = delegate(MyNode target)
            {
                if (target is MyWorkingNode)
                {
                    MyWorkingNode workingNode = target as MyWorkingNode;

                    if (!string.IsNullOrEmpty(workingNode.DataFolder))
                    {
                        Uri uri = new Uri(workingNode.DataFolder, UriKind.RelativeOrAbsolute);

                        if (!uri.IsAbsoluteUri)
                        {
                            workingNode.DataFolder = referencePath + "\\" + workingNode.DataFolder;
                        }
                    }
                }
            };

            convertAction(World);
            Network.Iterate(true, convertAction);
        }
        */

        #endregion

        #region Utility functins

        public static string ShortenMemoryBlockName(string name)
        {
            string result = name;

            int textEnd = name.LastIndexOf("Output");
            if (textEnd > 0)
            {
                result = result.Substring(0, textEnd);
            }

            textEnd = name.LastIndexOf("Input");
            if (textEnd > 0)
            {
                result = result.Substring(0, textEnd);
            }

            return result != String.Empty ? result : name;
        }

        public static string RemovePostfix(string name, string postFix)
        {
            string result = name;
            int textEnd = name.LastIndexOf(postFix);

            if (textEnd > 0)
            {
                result = result.Substring(0, textEnd);
            }

            return result != String.Empty ? result : name;
        }

        /// <summary>Strips "My" prefix (only when it is followed by a capital letter)</summary>
        public static string ShortenNodeTypeName(Type nodeType)
        {
            // only strip leading "My" if the name is "MyCapitalizedSomething"
            if ((nodeType.Name.Length > 2) && nodeType.Name.StartsWith("My") && Char.IsUpper(nodeType.Name[2]))
                return nodeType.Name.Substring(2);
            else
                return nodeType.Name;
        }

        #endregion

        public void Restore()
        {
            RestoreObservers();
            RestoreDashboard();
        }

        private void RestoreDashboard()
        {
            if (Dashboard == null)
                Dashboard = new Dashboard.Dashboard();

            if (GroupedDashboard == null)
                GroupedDashboard = new GroupDashboard();

            // The order is important - the normal dashboard properties must be set up
            // before they're added to groups.
            Dashboard.RestoreFromIds(this);
            GroupedDashboard.RestoreFromIds(this);
        }

        public void RestoreObservers()
        {
            if (Observers == null)
                return;

            foreach (MyAbstractObserver observer in Observers)
                observer.RestoreTargetFromIdentifier(this);
        }
    }
}
