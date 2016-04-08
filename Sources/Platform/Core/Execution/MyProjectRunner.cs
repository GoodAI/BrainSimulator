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
using GoodAI.TypeMapping;

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
* Open/save/run projects (OpenProject, SaveProject, RunAndPause, Shutdown)
* Edit node and task parameters (Set)
* Edit multiple nodes at once (Filter, GetNodesOfType)
* Get or set memory block values (GetValues, SetValues)
* Track specific memory block values, export data or plot them (TrackValue, Results, SaveResults, PlotResults)

## Simple example

    MyProjectRunner runner = new MyProjectRunner(MyLogLevel.DEBUG);
    runner.OpenProject(@"C:\Users\johndoe\Desktop\Breakout.brain");
    runner.DumpNodes();
    runner.RunAndPause(1000, 100);
    float[] data = runner.GetValues(24, "Output");
    MyLog.INFO.WriteLine(data);
    runner.Shutdown();

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
                runner.RunAndPause(1, 10);
                float okDot = runner.GetValues(8)[0];
                okSum += okDot;
                runner.Reset();
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
    runner.Shutdown();
*/

namespace GoodAI.Core.Execution
{
    public class MyProjectRunner : IDisposable
    {
        private MyProject m_project;

        public MySimulationHandler SimulationHandler { get; private set; }

        private int m_resultIdCounter;
        protected delegate float[] MonitorFunc(MySimulation simulation);
        private List<Tuple<int, uint, MonitorFunc>> m_monitors;
        private readonly Hashtable m_results;

        /// <summary>
        /// Definition for filtering function
        /// </summary>
        /// <param name="node">Node, which is being processed for filtering</param>
        /// <returns></returns>
        public delegate bool FilterFunc(MyNode node);

        public MyProject Project
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
                m_project.SimulationHandler = SimulationHandler;
            }
        }

        public uint SimulationStep
        {
            get { return SimulationHandler.SimulationStep; }
        }

        public MyProjectRunner(MyLogLevel level = MyLogLevel.DEBUG)
        {
            MySimulation simulation = TypeMap.GetInstance<MySimulation>();
            SimulationHandler = new MySimulationHandler(simulation);
            m_resultIdCounter = 0;

            SimulationHandler.ProgressChanged += SimulationHandler_ProgressChanged;
            SimulationHandler.StepPerformed += SimulationHandler_StepPerformed;

            m_monitors = new List<Tuple<int, uint, MonitorFunc>>();
            m_results = new Hashtable();

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
                throw;
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
        /// Filter all nodes in project recursively. Returns list of nodes, for which the filter function returned True.
        /// </summary>
        /// <param name="filterFunc">User-defined function for filtering</param>
        /// <returns>Node list</returns>
        public List<MyNode> Filter(FilterFunc filterFunc)
        {
            List<MyNode> nodes = new List<MyNode>();
            MyNodeGroup.IteratorAction a = x => { if (x.GetType() != typeof(MyNodeGroup) && filterFunc(x)) { nodes.Add(x); } };
            Project.Network.Iterate(true, a);
            return nodes;
        }

        /// <summary>
        /// Returns list of nodes of given type
        /// </summary>
        /// <param name="type">Node type</param>
        /// <returns>Node list</returns>
        public List<MyNode> GetNodesOfType(Type type)
        {
            // Comparing strings is a really really bad hack. But == comparison of GetType() (or typeof) and type or using Equals methods stopped working.
            // And you cannot use "is" since the "type" is known at the execution time
            FilterFunc filter = (x => x.GetType().ToString() == type.ToString());
            return Filter(filter);
        }

        /// <summary>
        /// Return task of given type from given node
        /// </summary>
        /// <param name="node">Node</param>
        /// <param name="type">Type of task</param>
        /// <returns>Task</returns>
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
        /// <param name="nodeId">Node ID</param>
        /// <param name="blockName">Memory block name</param>
        /// <returns>Retrieved values</returns>
        public float[] GetValues(int nodeId, string blockName = "Output")
        {
            MyMemoryBlock<float> block = GetMemBlock(nodeId, blockName);
            block.SafeCopyToHost();
            return block.Host;
        }

        /// <summary>
        /// Set values of memory block
        /// </summary>
        /// <param name="nodeId">Node ID</param>
        /// <param name="values">Values to be set</param>
        /// <param name="blockName">Name of memory block</param>
        public void SetValues(int nodeId, float[] values, string blockName = "Input")
        {
            MyLog.INFO.WriteLine("Setting values of " + blockName + "@" + nodeId);
            MyMemoryBlock<float> block = GetMemBlock(nodeId, blockName);
            for (int i = 0; i < block.Count; ++i)
            {
                block.Host[i] = values[i];
            }
            block.SafeCopyToDevice();
        }

        protected MyWorkingNode GetNode(int nodeId)
        {
            return Project.GetNodeById(nodeId) as MyWorkingNode;
        }

        public void SaveOnStop(int nodeId, bool shallSave)
        {
            GetNode(nodeId).SaveOnStop = shallSave;
            MyLog.INFO.WriteLine("Save@" + nodeId + " set to " + shallSave);
        }

        public void LoadOnStart(int nodeId, bool shallLoad)
        {
            GetNode(nodeId).LoadOnStart = shallLoad;
            MyLog.INFO.WriteLine("Load@" + nodeId + " set to " + shallLoad);
        }

        /// <summary>
        /// Shutdown the runner and the underlaying simulation infrastructure
        /// </summary>
        public void Shutdown()
        {
            SimulationHandler.Finish();
        }

        public void Dispose()
        {
            Shutdown();
        }

        /// <summary>
        /// Loads project from file
        /// </summary>
        /// <param name="path">Path to .brain/.brainz file</param>
        public void OpenProject(string path)
        {
            MyLog.INFO.WriteLine("Loading project: " + path);

            string content;

            try
            {
                string newProjectName = MyProject.MakeNameFromPath(path);

                content = ProjectLoader.LoadProject(path,
                    MyMemoryBlockSerializer.GetTempStorage(newProjectName));

                using (MyMemoryManager.Backup backup = MyMemoryManager.GetBackup())
                {
                    Project = MyProject.Deserialize(content, Path.GetDirectoryName(path));
                    backup.Forget();
                }

                Project.Name = newProjectName;
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project loading failed: " + e.Message);
                throw;
            }
        }

        /// <summary>
        /// Saves project to given path
        /// </summary>
        /// <param name="path">Path for saving .brain/.brainz file</param>
        public void SaveProject(string path)
        {
            MyLog.INFO.WriteLine("Saving project: " + path);
            try
            {
                string fileContent = Project.Serialize(Path.GetDirectoryName(path));
                ProjectLoader.SaveProject(path, fileContent,
                    MyMemoryBlockSerializer.GetTempStorage(Project));
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Project saving failed: " + e.Message);
                throw;
            }
        }

        protected void SetProperty(object o, string propName, object value)
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
        /// Sets property of given node. Support Enums - enter enum value as a string
        /// </summary>
        /// <param name="nodeId">Node ID</param>
        /// <param name="propName">Property name</param>
        /// <param name="value">Value to be set</param>
        public void Set(int nodeId, string propName, object value)
        {
            MyNode n = Project.GetNodeById(nodeId);
            SetProperty(n, propName, value);
        }

        /// <summary>
        /// Sets property of given task. Support Enums
        /// </summary>
        /// <param name="nodeId">Node ID</param>
        /// <param name="taskType">Task type</param>
        /// <param name="propName">Property name</param>
        /// <param name="value">New property value</param>
        public void Set(int nodeId, Type taskType, string propName, object value)
        {
            MyWorkingNode node = (Project.GetNodeById(nodeId) as MyWorkingNode);
            MyTask task = GetTaskByType(node, taskType);
            SetProperty(task, propName, value);
        }

        /// <summary>
        /// Track a value
        /// </summary>
        /// <param name="nodeId">Node ID</param>
        /// <param name="blockName">Memory block name</param>
        /// <param name="blockOffset">Offset in given memory block</param>
        /// <param name="trackInterval">Track value each x steps</param>
        /// <returns>Result ID</returns>
        public int TrackValue(int nodeId, string blockName = "Output", int blockOffset = 0, uint trackInterval = 10)
        {
            MonitorFunc valueMonitor = x =>
            {
                float value = GetValues(nodeId, blockName)[blockOffset];
                float[] record = new float[2] { SimulationHandler.SimulationStep, value };
                return record;
            };

            Tuple<int, uint, MonitorFunc> rec = new Tuple<int, uint, MonitorFunc>(m_resultIdCounter++, trackInterval, valueMonitor);
            m_monitors.Add(rec);
            m_results[rec.Item1] = new List<float[]>();
            MyLog.INFO.WriteLine(blockName + "[" + blockOffset + "]@" + nodeId + "is now being tracked with ID " + (m_resultIdCounter - 1));
            return m_resultIdCounter - 1;
        }

        /// <summary>
        /// Returns hashtable with results (list of float arrays)
        /// </summary>
        /// <returns>Results</returns>
        public Hashtable Results()
        {
            return m_results;
        }

        /// <summary>
        /// Save result to a file
        /// </summary>
        /// <param name="resultId">Result ID</param>
        /// <param name="outputPath">Path to file in C# format, e.g. C:\path\to\file</param>
        public void SaveResults(int resultId, string outputPath)
        {
            string data = "";
            foreach (float[] x in (Results()[resultId] as List<float[]>))
            {
                data += x[0] + " " + x[1] + "\r\n";
            }

            System.IO.StreamWriter file = new System.IO.StreamWriter(outputPath);
            file.WriteLine(data);
            file.Close();
            MyLog.INFO.WriteLine("Results saved to " + outputPath);
        }

        /// <summary>
        /// Plot results to a file
        /// </summary>
        /// <param name="resultIds">IDs of results</param>
        /// <param name="outputPath">Path to file in gnuplot format, e.g. C:/path/to/file</param>
        /// <param name="lineTitles">Titles of the lines</param>
        /// <param name="dimensionSizes">Sizes of plot dimensions</param>
        public void PlotResults(int[] resultIds, string outputPath, string[] lineTitles = null, int[] dimensionSizes = null)
        {
            string data = "";
            string args = "-e \"set term png";
            if (dimensionSizes != null)
            {
                args += " size " + dimensionSizes[0] + "," + dimensionSizes[1];
            }
            args += "; set output '" + outputPath + "'; plot";
            for (int i = 0; i < resultIds.Length; ++i)
            {
                args += "'-' using 1:2 ";
                if (lineTitles != null)
                {
                    string title = lineTitles[i];
                    args += "title \\\"" + title + "\\\" ";
                }
                args += "smooth frequency";
                if (i == resultIds.Length - 1)
                {
                    args += ";";
                }
                else
                {
                    args += ", ";
                }

                foreach (float[] x in (Results()[resultIds[i]] as List<float[]>))
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
            MyLog.INFO.WriteLine("Results " + resultIds + " plotted to " + outputPath);
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
            foreach (Tuple<int, uint, MonitorFunc> m in m_monitors)
            {
                if (SimulationHandler.SimulationStep % m.Item2 == 0)
                {
                    float[] value = (float[])m.Item3(SimulationHandler.Simulation);
                    (m_results[m.Item1] as List<float[]>).Add(value);
                }
            }
        }

        /// <summary>
        /// Runs simulation for a given number of steps. Simulation will be left in PAUSED state after
        /// this function returns, allowing to inspect content of memory blocks and then perhaps
        /// resume the simulation by calling this function again.
        /// </summary>
        /// <param name="stepCount">Number of steps to perform</param>
        /// <param name="reportInterval">Step count between printing out simulation info (e.g. speed)</param>
        public void RunAndPause(uint stepCount, uint reportInterval = 100)
        {
            if (stepCount == 0)
                throw new ArgumentException("Zero step count not allowed.", "stepCount");  // would run forever
            // TODO(Premek): Add a check that that simulation is not finished

            if (SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
            {
                if (SimulationHandler.UpdateMemoryModel())
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Memory model did not converge.");
                    return;
                }

                SimulationHandler.Simulation.Validate(Project);

                MyValidator validator = SimulationHandler.Simulation.Validator;

                if (!validator.ValidationSucessfull)
                {
                    MyLog.ERROR.WriteLine("Simulation cannot be started! Validation failed.");
                    return;
                }
            }

            try
            {
                SimulationHandler.ReportIntervalSteps = reportInterval;
                SimulationHandler.StartSimulation(stepCount);
                SimulationHandler.WaitUntilStepsPerformed();
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Simulation cannot be started! Exception occured: " + e.Message);
                throw;
            }
        }

        /// <summary>
        /// Stops the paused simulation and flushes memory
        /// </summary>
        public void Reset()
        {
            SimulationHandler.StopSimulation();
            SimulationHandler.Simulation.ResetSimulationStep();  // reset simulation step back to 0
            m_monitors.Clear();
            m_results.Clear();
            m_resultIdCounter = 0;
        }

        public MyProject CreateProject(Type worldType, string projectName = null)
        {
            Project = new MyProject
            {
                Network = new MyNetwork(),
                Name = projectName
            };

            Project.CreateWorld(worldType);

            return Project;
        }
    }
}
