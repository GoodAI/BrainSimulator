using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Project;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Diagnostics;

/*
# How to use MyProjectRunner

MyProjectRunner is an alternative to BrainSimulator GUI - you can use it to operate with projects without using GUI. 
That is especially handy if you want to:
* Edit 10s or 100s of nodes at once
* Try multiple values for one parameter and watch how it will affect results
* Run project for specific number of steps - then make some changes - run project again. And again
* Create multiple versions of one project
* Get information about value changes during the simulation
* Increase speed of simulation (no GUI)

## Current capabilities
* Print info (DumpNodes)
* Open/save/run projects (OpenProject, SaveProject, Run, Quit)
* Edit node and task parameters (Set)
* Edit multiple nodes at once (Filter, GetNodesOfType)
* Get or set memory block values (GetValues, SetValues)
* Track specific memory block values, export data or plot them (TrackValue, Results, SaveResults, PlotResults)

## Simple example

    MyProjectRunner runner = new MyProjectRunner(MyLogLevel.DEBUG);
    runner.OpenProject(@"C:\Users\johndoe\Desktop\Breakout.brain");
    runner.DumpNodes();
    runner.Run(1000, 100);
    float[] data = runner.GetValues(24, "Output");
    MyLog.INFO.WriteLine(data);
    runner.Quit();
    Console.ReadLine();

## More advanced example
    
    // Program tries different combinations of parameters for two nodes, computes average values for multiple runs, log results and saves them to file.

    MyProjectRunner runner = new MyProjectRunner(MyLogLevel.WARNING);
    runner.OpenProject(@"C:\Users\johndoe\Desktop\test.brain");
    float iterations = 250;

    List<Tuple<int, int, float, float>> results = new List<Tuple<int, int, float, float>>();
    runner.Set(6, "OutputSize", 32);

    for (int symbolSize = 512; symbolSize <= 8192; symbolSize *= 2)
    {
       for (int binds = 20; binds <= 50; binds += 5)
       {
            float okSum = 0;
            runner.Set(7, "Binds", binds);
            runner.Set(7, "SymbolSize", symbolSize);
            for (int i = 0; i < iterations; ++i)
            {
                runner.Run(1, 10);
                float okDot = runner.GetValues(8)[0];
                okSum += okDot;
                runner.Stop();
                if ((i + 1) % 10 == 0)
                {
                    MyLog.WARNING.Write('.');
                }
            }
            MyLog.WARNING.WriteLine();
            float wrongSum = 1;
            MyLog.WARNING.WriteLine("Results:" + symbolSize + "@" + binds + " => " + okSum / iterations + " / " + wrongSum / iterations);
            results.Add(new Tuple<int, int, float, float>(symbolSize, binds, okSum / iterations, wrongSum / iterations));
        }
    }

    File.WriteAllLines(@"C:\Users\johndoe\Desktop\results.txt", results.Select(n => n.ToString().Substring(1, n.ToString().Length - 2)));
    runner.Quit();
*/

namespace GoodAI.Core.Execution
{
    public class MyProjectRunner
    {
        private static MyProject m_project;

        public static MySimulationHandler SimulationHandler { get; private set; }

        public static List<Tuple<int, uint, MonitorFunc>> SimulationMonitors { get; private set; }
        public static Hashtable MonitorResults;
        public delegate float[] MonitorFunc(MySimulation simulation);

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

        public MyProjectRunner(MyLogLevel level = MyLogLevel.DEBUG)
        {
            MySimulation simulation = new MyLocalSimulation();
            SimulationHandler = new MySimulationHandler(simulation);
            uid = 0;

            SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;
            SimulationHandler.StepPerformed += SimulationHandler_StepPerformed;

            SimulationMonitors = new List<Tuple<int, uint, MonitorFunc>>();
            MonitorResults = new Hashtable();

            Project = new MyProject();

            var path = MyResources.GetEntryAssemblyPath();

            if (MyConfiguration.ModulesSearchPath.Count == 0)
                MyConfiguration.SetupModuleSearchPath();
            MyConfiguration.ProcessCommandParams();

            try
            {
                if (MyConfiguration.Modules.Count == 0)
                    MyConfiguration.LoadModules();
            }
            catch (Exception e)
            {
                MyLog.WARNING.WriteLine(e.Message);
                Environment.Exit(1);
            }

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
            // Comparing strings is a really really bad hack. But == comparison of GetType() (or typeof) and type or using Equals methods stopped working.
            // And you cannot use "is" since the "type" is known at the execution time
            FilterFunc filter = x => { if (x.GetType().ToString() == type.ToString()) return true; return false; };
            return Filter(filter);
        }

        /// <summary>
        /// Return task of given type from given node
        /// </summary>
        /// <param name="node">Node</param>
        /// <param name="type">Type of task</param>
        /// <returns></returns>
        protected MyTask GetTaskByType(MyWorkingNode node, Type type)
        {
            foreach (PropertyInfo taskPropInfo in node.GetInfo().TaskOrder)
            {
                MyTask task = node.GetTaskByPropertyName(taskPropInfo.Name);
                if (task.GetType().ToString() == type.ToString())
                    return task;
            }
            return null;
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
            SimulationHandler.Finish();
        }

        /// <summary>
        /// Loads project from file
        /// </summary>
        /// <param name="fileName">Path to .brain file</param>
        public void OpenProject(string fileName)
        {
            MyLog.INFO.WriteLine("Loading project: " + fileName);

            string content;

            try
            {
                string newProjectName = Path.GetFileNameWithoutExtension(fileName);

                content = ProjectLoader.LoadProject(fileName,
                    MyMemoryBlockSerializer.GetTempStorage(newProjectName));

                Project = MyProject.Deserialize(content, Path.GetDirectoryName(fileName));
                Project.Name = newProjectName;
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
                //Project.Name = Path.GetFileNameWithoutExtension(fileName);  // a little sideeffect (should be harmless)

                string fileContent = Project.Serialize(Path.GetDirectoryName(fileName));
                ProjectLoader.SaveProject(fileName, fileContent,
                    MyMemoryBlockSerializer.GetTempStorage(Project));
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
            MyTask task = GetTaskByType(node, taskType);
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
            MonitorFunc valueMonitor = x =>
            {
                float value = GetValues(id, memName)[offset];
                float[] record = new float[2] { SimulationHandler.SimulationStep, value };
                return record;
            };

            Tuple<int, uint, MonitorFunc> rec = new Tuple<int, uint, MonitorFunc>(uid++, step, valueMonitor);
            SimulationMonitors.Add(rec);
            MonitorResults[rec.Item1] = new List<float[]>();
            MyLog.INFO.WriteLine(memName + "[" + offset + "]@" + id + "is now being tracked with ID " + (uid - 1));
            return uid - 1;
        }

        /// <summary>
        /// Returns hashtable with reults (list of float arrays)
        /// </summary>
        /// <returns>Results</returns>
        public Hashtable Results()
        {
            return MonitorResults;
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

        void SimulationHandler_ProgressChanged(object sender, System.ComponentModel.ProgressChangedEventArgs e)
        {
            if (SimulationHandler.State != MySimulationHandler.SimulationState.STOPPED)
            {
                MyLog.INFO.WriteLine("[" + SimulationHandler.SimulationStep + "] Running at " + SimulationHandler.SimulationSpeed + "/s");
            }
        }

        void SimulationHandler_StepPerformed(object sender, System.ComponentModel.ProgressChangedEventArgs e)
        {
            foreach (Tuple<int, uint, MonitorFunc> m in SimulationMonitors)
            {
                if (SimulationHandler.SimulationStep % m.Item2 == 0)
                {
                    float[] value = (float[])m.Item3(SimulationHandler.Simulation);
                    (MonitorResults[m.Item1] as List<float[]>).Add(value);
                }
            }
        }

        /// <summary>
        /// Runs simulation for given number of steps
        /// </summary>
        /// <param name="steps">Number of steps</param>
        /// <param name="logStep">Print info each x steps</param>
        public void Run(uint steps, uint logStep = 100)
        {
            if (SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
            {
                if (SimulationHandler.UpdateMemoryModel())
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Memory model did not converge.");
                    return;
                }

                MyValidator validator = new MyValidator();
                validator.Simulation = SimulationHandler.Simulation;
                validator.ClearValidation();

                Project.World.ValidateWorld(validator);
                Project.Network.Validate(validator);

                validator.Simulation = null;

                if (!validator.ValidationSucessfull)
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Validation failed.");
                    return;
                }
            }
            try
            {
                SimulationHandler.ReportIntervalSteps = logStep;
                SimulationHandler.StartSimulation(true, steps);
                SimulationHandler.WaitUntilStepsPerformed();
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Simulation cannot be started! Exception occured: " + e.Message);
            }
        }

        public void Stop()
        {
            SimulationHandler.StopSimulation();
            SimulationMonitors.Clear();
            MonitorResults.Clear();
            uid = 0;
        }
    }
}
