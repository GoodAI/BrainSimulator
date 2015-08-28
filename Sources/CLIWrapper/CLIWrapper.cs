using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace CLIWrapper
{
    public class BSCLI
    {
        private static MyProject m_project;
        private static int MAX_BLOCKS_UPDATE_ATTEMPTS = 20;

        public static MyCLISimulationHandler SimulationHandler { get; private set; }

        /// <summary>
        /// Definition for filtering function
        /// </summary>
        /// <param name="node">Node, which is being processed for filtering</param>
        /// <returns></returns>
        public delegate bool FilterFunc(MyNode node);

        private int uid;

        public static MyProject Project
        {
            get { return m_project; }
            private set
            {
                if (m_project != null)
                {
                    m_project.Dispose();
                }
                m_project = value;
                SimulationHandler.Project = value;
            }
        }

        public static void Main(string[] args)
        {
            BSCLI cli = new BSCLI();
            cli.OpenProject(@"C:\Users\michal.vlasak\Downloads\lt.brain");
            //List<MyNode> nodes = CLI.GetNodesOfType(typeof(MyCodeBook));
            //CLI.DumpNodes(); // OK

            //CLI.Set(323, "Mode", "GenerateOutput");  // OK
            //CLI.Set(323, "InputAsOffset", true); // OK
            //CLI.Set(323, "SymbolSize", 65); // OK

            //BrainSimulatorCLI.BSCLI.FilterFunc filter = x => { return true; };

            //float[] values = CLI.GetValues(527, "Output");
            //foreach (float f in values) {
            //    MyLog.DEBUG.WriteLine(f);
            //}
            cli.Run(100);
            //CLI.SaveProject(@"C:\Users\michal.vlasak\Desktop\export.brain");
            cli.Quit();
            Console.ReadLine();
        }

        public BSCLI(MyLogLevel level = MyLogLevel.DEBUG)
        {
            SimulationHandler = new MyCLISimulationHandler();
            SimulationHandler.Simulation = new MyLocalSimulation();
            uid = 0;

            MyConfiguration.LoadModules();
            //MyConfiguration.AddNodesFromFile(@"conf\basic_nodes.xml", Assembly.Load(new AssemblyName("BrainSimulator")));
            //MyConfiguration.AddNodesFromFile(@"conf\custom_nodes.xml", Assembly.Load(new AssemblyName("CustomModels")));

            MyLog.Level = level;
        }

        /// <summary>
        /// Prints info about nodes to DEBUG
        /// </summary>
        public void DumpNodes()
        {
            MyNodeGroup.IteratorAction a = x => { MyLog.DEBUG.WriteLine("[{0}] {1}: {2}", x.Id, x.Name, x.GetType()); };
            Project.Network.Iterate(true, a);
        }

        /// <summary>
        /// Not implemented yet
        /// </summary>
        public void DumpConnections()
        {
            throw new NotImplementedException();
            /*foreach (MyConnectionProxy c in Project.Network.m_connections) { 
            
            }*/
        }

        /// <summary>
        /// Filter all nodes in project recursively. Returns list of nodes, for which the filter function returned True
        /// </summary>
        /// <param name="userFunc">User-defined function for filtering</param>
        /// <returns></returns>
        public List<MyNode> Filter(FilterFunc userFunc)
        {
            List<MyNode> nodes = new List<MyNode>();
            MyNodeGroup.IteratorAction a = x => { if (x.GetType() != typeof(MyNodeGroup) && userFunc(x)) { nodes.Add(x); } };
            Project.Network.Iterate(true, a);
            return nodes;
        }

        /// <summary>
        /// Returns list of nodes of given type. Uses Filter function
        /// </summary>
        /// <param name="type">Node type</param>
        /// <returns></returns>
        public List<MyNode> GetNodesOfType(Type type)
        {
            FilterFunc filter = x => { if (x.GetType() == type) return true; return false; };
            return Filter(filter);
        }

        protected MyMemoryBlock<float> GetMemBlock(int nodeId, string blockName)
        {
            MyNode n = Project.GetNodeById(nodeId);
            PropertyInfo mem = n.GetType().GetProperty(blockName);
            MyMemoryBlock<float> block = mem.GetValue(n, null) as MyMemoryBlock<float>;
            return block;
        }

        /// <summary>
        /// Returns float array of value from memory block of given node
        /// </summary>
        /// <param name="id">Node ID</param>
        /// <param name="blockName">Memory block name</param>
        /// <returns></returns>
        public float[] GetValues(int id, string blockName = "Output")
        {
            MyMemoryBlock<float> block = GetMemBlock(id, blockName);
            block.SafeCopyToHost();
            return block.Host as float[];
        }

        /// <summary>
        /// Set values of memory block
        /// </summary>
        /// <param name="id">Node ID</param>
        /// <param name="blockName">Name of memory block</param>
        /// <param name="values">Values to be set</param>
        public void SetValues(int id, float[] values, string blockName = "Input")
        {
            MyLog.INFO.WriteLine("Setting values of " + blockName + "@" + id);
            MyMemoryBlock<float> block = GetMemBlock(id, blockName);
            for (int i = 0; i < block.Count; ++i)
            {
                block.Host[i] = values[i];
            }
            block.SafeCopyToDevice();
        }

        protected MyWorkingNode Get(int id)
        {
            return Project.GetNodeById(id) as MyWorkingNode;
        }

        public void Save(int id, bool b)
        {
            Get(id).SaveOnStop = b;
            MyLog.INFO.WriteLine("Save@" + id + " set to " + b);
        }

        public void Load(int id, bool b)
        {
            Get(id).LoadOnStart = b;
            MyLog.INFO.WriteLine("Load@" + id + " set to " + b);
        }

        /// <summary>
        /// Quits the simulation
        /// </summary>
        public void Quit()
        {
            if (SimulationHandler.State != MyCLISimulationHandler.SimulationState.STOPPED)
            {
                SimulationHandler.StopSimulation();
            }
            while (SimulationHandler.State != MyCLISimulationHandler.SimulationState.STOPPED) { }
            SimulationHandler.Finish();
        }

        /// <summary>
        /// Loads project from file
        /// </summary>
        /// <param name="fileName">Path to .brain file</param>
        public void OpenProject(string fileName)
        {
            MyLog.INFO.WriteLine("Loading project: " + fileName);

            try
            {
                TextReader reader = new StreamReader(fileName);
                string content = reader.ReadToEnd();
                reader.Close();

                Project = MyProject.Deserialize(content, Path.GetDirectoryName(fileName));
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project loading failed: " + e.Message);
            }

        }

        /// <summary>
        /// Saves project to given path
        /// </summary>
        /// <param name="fileName">Path for saving</param>
        public void SaveProject(string fileName)
        {
            MyLog.INFO.WriteLine("Saving project: " + fileName);
            try
            {
                Project.Name = Path.GetFileNameWithoutExtension(fileName);
                string fileContent = Project.Serialize(Path.GetFileNameWithoutExtension(fileName));

                TextWriter writer = new StreamWriter(fileName);
                writer.Write(fileContent);
                writer.Close();
                Project.Observers = null;
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project saving failed: " + e.Message);
            }
        }

        /// <summary>
        /// Sets property of given object
        /// </summary>
        /// <param name="o">Object</param>
        /// <param name="propName">Name of property</param>
        /// <param name="value">Value of property</param>
        protected void setProperty(object o, string propName, object value)
        {
            MyLog.DEBUG.WriteLine("Setting property " + propName + "@" + o + " to " + value);
            PropertyInfo pInfo = o.GetType().GetProperty(propName);
            Type pType = pInfo.PropertyType;
            if (pType.IsEnum)
            {
                if (Enum.IsDefined(pType, value))
                {
                    object a = Enum.Parse(pType, (string)value);
                    pInfo.SetValue(o, a);
                }
            }
            else
            {
                pInfo.SetValue(o, Convert.ChangeType(value, pType));
            }
        }

        /// <summary>
        /// Sets property of given node. Support Enums
        /// </summary>
        /// <param name="id">Node ID</param>
        /// <param name="propName">Property name</param>
        /// <param name="value">Value to be set</param>
        public void Set(int id, string propName, object value)
        {
            MyNode n = Project.GetNodeById(id);
            setProperty(n, propName, value);
        }

        /// <summary>
        /// Sets property of given task. Support Enums
        /// </summary>
        /// <param name="id">Node ID</param>
        /// <param name="taskType">Task type</param>
        /// <param name="propName">Property name</param>
        /// <param name="value">New property value</param>
        public void Set(int id, Type taskType, string propName, object value)
        {
            MyWorkingNode node = (Project.GetNodeById(id) as MyWorkingNode);
            MyTask task = (Project.GetNodeById(id) as MyWorkingNode).GetTaskByType(taskType);
            setProperty(task, propName, value);
        }

        /// <summary>
        /// Track a value
        /// </summary>
        /// <param name="id">Node ID</param>
        /// <param name="length">How many steps to track</param>
        /// <param name="offset">Offset in given memory block</param>
        /// <param name="memName">Memory block name</param>
        /// <param name="step">Track value each x steps</param>
        /// <returns>Id of result</returns>
        public int TrackValue(int id, int length, uint step = 10, int offset = 0, string memName = "Output")
        {
            MyCLISimulationHandler.MonitorFunc valueMonitor = x =>
            {
                float value = GetValues(id, memName)[offset];
                float[] record = new float[2] { SimulationHandler.SimulationStep, value };
                return record;
            };

            Tuple<int, uint, MyCLISimulationHandler.MonitorFunc> rec = new Tuple<int, uint, MyCLISimulationHandler.MonitorFunc>(uid++, step, valueMonitor);
            SimulationHandler.AddMonitor(rec);
            MyLog.INFO.WriteLine(memName + "[" + offset + "]@" + id + "is now being tracked with ID " + (uid - 1));
            return uid - 1;
        }

        /// <summary>
        /// Returns hashtable with reults (list of float arrays)
        /// </summary>
        /// <returns>Results</returns>
        public Hashtable Results()
        {
            return SimulationHandler.Results();
        }

        /// <summary>
        /// Save results to a file
        /// </summary>
        /// <param name="id">ID of results</param>
        /// <param name="output">Path to file in C# format, e.g. C:\path\to\file</param>
        public void SaveResults(int id, string output)
        {
            string data = "";
            foreach (float[] x in (Results()[id] as List<float[]>))
            {
                data += x[0] + " " + x[1] + "\r\n";
            }

            System.IO.StreamWriter file = new System.IO.StreamWriter(output);
            file.WriteLine(data);
            file.Close();
            MyLog.INFO.WriteLine("Results saved to " + output);
        }

        /// <summary>
        /// Plot results to a file
        /// </summary>
        /// <param name="ids">IDs of results</param>
        /// <param name="output">Path to file in gnuplot format, e.g. C:/path/to/file</param>
        /// <param name="titles">Titles of the lines</param>
        public void PlotResults(int[] ids, string output, string[] titles = null, int[] dims = null)
        {
            string data = "";
            string args = "-e \"set term png";
            if (dims != null)
            {
                args += " size " + dims[0] + "," + dims[1];
            }
            args += "; set output '" + output + "'; plot";
            for (int i = 0; i < ids.Length; ++i)
            {
                args += "'-' using 1:2 ";
                if (titles != null)
                {
                    string title = titles[i];
                    args += "title \\\"" + title + "\\\" ";
                }
                args += "smooth frequency";
                if (i == ids.Length - 1)
                {
                    args += ";";
                }
                else
                {
                    args += ", ";
                }

                foreach (float[] x in (Results()[ids[i]] as List<float[]>))
                {
                    data += x[0] + " " + x[1] + "\n";
                }
                data += "e\n";
            }
            args += "\"";
            //This is needed for redirecting STDIN which is needed for inputing data to gnuplot without external file - otherwise simple Process.Start() would do
            Process p = new Process();
            p.StartInfo.FileName = "gnuplot";
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardInput = true;
            p.StartInfo.Arguments = args;
            p.Start();
            p.StandardInput.WriteLine(data);
            MyLog.INFO.WriteLine("Results " + ids + " plotted to " + output);
        }

        private bool UpdateAndCheckChange(MyNode node)
        {
            node.PushOutputBlockSizes();
            node.UpdateMemoryBlocks();
            return node.AnyOutputSizeChanged();
        }

        /// <summary>
        /// Runs simulation for given number of steps
        /// </summary>
        /// <param name="steps">Number of steps</param>
        /// <param name="logStep">Print info each x steps</param>
        public void Run(uint steps, uint logStep = 100)
        {
            if (SimulationHandler.State == MyCLISimulationHandler.SimulationState.STOPPED)
            {
                MyLog.INFO.WriteLine("--------------");
                MyLog.INFO.WriteLine("Updating memory blocks...");

                IMyOrderingAlgorithm topoOps = new MyHierarchicalOrdering();
                List<MyNode> orderedNodes = topoOps.EvaluateOrder(Project.Network);

                if (!orderedNodes.Any())
                {
                    return;
                }

                int attempts = 0;
                bool anyOutputChanged = false;

                try
                {
                    while (attempts < MAX_BLOCKS_UPDATE_ATTEMPTS)
                    {
                        attempts++;
                        anyOutputChanged = false;

                        anyOutputChanged |= UpdateAndCheckChange(Project.World);
                        orderedNodes.ForEach(node => anyOutputChanged |= UpdateAndCheckChange(node));

                        if (!anyOutputChanged)
                        {
                            MyLog.INFO.WriteLine("Successful update after " + attempts + " cycle(s).");
                            break;
                        }
                    }
                }
                catch (Exception e)
                {
                    MyLog.ERROR.WriteLine("Exception occured while updating memory model: " + e.Message);
                    return;
                }

                /*MyValidator validator = ValidationView.Validator;
                validator.Simulation = SimulationHandler.Simulation;
                validator.ClearValidation();

                Project.World.ValidateWorld(validator);
                Project.Network.Validate(validator);
                validator.AssertError(!anyOutputChanged, Project.Network, "Possible infinite loop in memory block sizes.");

                ValidationView.UpdateListView();
                validator.Simulation = null;*/

                //if (validator.ValidationSucessfull)
                if (!true)
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Validation failed.");
                }
            }
            try
            {
                SimulationHandler.StartSimulation(steps, logStep);
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Simulation cannot be started! Exception occured: " + e.Message);
            }
        }

        public void Stop()
        {
            SimulationHandler.StopSimulation();
            uid = 0;
        }
    }
}
