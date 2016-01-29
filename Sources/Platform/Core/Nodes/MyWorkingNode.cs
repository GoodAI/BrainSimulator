using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reflection;
using GoodAI.Core.Execution;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    public interface IMyCustomTaskFactory
    {
        void CreateTasks();
    }

    /// <summary>
    /// This encapsulates a task group.
    /// </summary>
    public class TaskGroup
    {
        public string GroupName { get; private set; }
        public MyWorkingNode Owner { get; private set; }

        public TaskGroup(MyWorkingNode owner, string groupName)
        {
            GroupName = groupName;
            Owner = owner;
            Tasks = new List<MyTask>();
        }

        public List<MyTask> Tasks { get; private set; }

        public MyTask GetTaskByName(string value)
        {
            return Tasks.FirstOrDefault(task => task.Name == value);
        }

        public void DisableAll()
        {
            foreach (MyTask task in Tasks.Where(task => task.Enabled))
                task.Enabled = false;
        }
    }

    public abstract class MyWorkingNode : MyNode
    {

        public virtual void Cleanup()
        {

        }

        #region Persistance

        [YAXSerializableField, YAXAttributeForClass]
        [MyBrowsable, Category("\t\tPersistance"), ReadOnly(true)]
        public bool LoadOnStart { get; set; }

        [YAXSerializableField, YAXAttributeForClass]
        [MyBrowsable, Category("\t\tPersistance"), ReadOnly(true)]
        public bool SaveOnStop { get; set; }

        [YAXSerializableField(DefaultValue = ""), YAXCustomSerializer(typeof(MyPathSerializer))]
        [MyBrowsable, Category("\t\tPersistance"), Editor]        
        public string DataFolder { get; set; }
        #endregion

        #region Node output

        protected MyAbstractMemoryBlock[] m_outputs;

        public override int OutputBranches
        {
            get { return base.OutputBranches; }
            set
            {
                base.OutputBranches = value;
                m_outputs = new MyAbstractMemoryBlock[value];
            }
        }

        public override MyMemoryBlock<float> GetOutput(int index)
        {
            return m_outputs.Length > index ? m_outputs[index] as MyMemoryBlock<float> : null;                        
        }

        public override MyMemoryBlock<T> GetOutput<T>(int index)
        {            
            return m_outputs.Length > index ? m_outputs[index] as MyMemoryBlock<T> : null;            
        }

        public override MyAbstractMemoryBlock GetAbstractOutput(int index)
        {
            return m_outputs.Length > index ? m_outputs[index] : null;
        }

        protected void SetOutput(int index, MyMemoryBlock<float> output)
        {
            if (m_outputs.Length > index)
            {
                m_outputs[index] = output;
            }
        }

        protected void SetOutput<T>(int index, MyMemoryBlock<T> output) where T : struct
        {
            if (m_outputs.Length > index)
            {
                m_outputs[index] = output;
            }
        }
        
        #endregion               
       
        #region Tasks        

        [YAXSerializableField, YAXSerializeAs("Tasks"), YAXCustomSerializer(typeof(MyTaskSerializer))]
        protected Dictionary<string, MyTask> m_tasks;

        public MyTask GetTaskByPropertyName(string propertyName)
        {            
            return m_tasks.ContainsKey(propertyName) ? m_tasks[propertyName] : null;
        }

        internal bool DisableTaskGroup(MyTask excludedTask)
        {
            bool anyTaskDisabled = false;
            
            if (!string.IsNullOrEmpty(excludedTask.TaskGroupName))
            {
                foreach (PropertyInfo pInfo in GetInfo().TaskGroups[excludedTask.TaskGroupName])
                {
                    MyTask task = GetTaskByPropertyName(pInfo.Name);
                    
                    if (task != excludedTask)
                    {
                        task.Enabled = false;
                        anyTaskDisabled = true;
                    }
                }                
            }

            return anyTaskDisabled;
        }

        public MyTask GetEnabledTask(string taskGroupName)
        {
            foreach (PropertyInfo pInfo in GetInfo().TaskGroups[taskGroupName])
            {
                MyTask task = GetTaskByPropertyName(pInfo.Name);

                if (task != null && task.Enabled)
                {
                    return task;
                }
            }
            return null;
        }

        internal void InitTasks()
        {
            foreach (MyTask task in m_tasks.Values)
            {
                task.Init(GPU);
            }
        }

        public void EnableFirstTask()
        {
            if (m_tasks.Count > 0)
            {
                (GetInfo().TaskOrder[0].GetValue(this) as MyTask).Enabled = true;                
            }
        }   
     
        public void EnableDefaultTasks()
        {
            foreach (MyTask task in m_tasks.Values.Reverse())
            {
                task.Enabled = task.EnabledByDefault;
            }
        }

        public void EnableAllTasks()
        {            
            foreach (MyTask task in m_tasks.Values.Reverse())
            {
                task.Enabled = true;
            }
        }        

        private void FinalizeTaskCreation()
        {
            foreach (PropertyInfo taskProperty in GetInfo().KnownTasks.Values)
            {
                MyTask task = taskProperty.GetValue(this) as MyTask;

                try
                {
                    if (task == null)
                    {
                        task = (MyTask)Activator.CreateInstance(taskProperty.PropertyType);
                        taskProperty.SetValue(this, task);
                    }

                    try
                    {
                        task.InitPropertiesDefault();
                    }
                    catch (Exception e)
                    {
                        MyLog.ERROR.WriteLine("Task initialization: " + e.Message);
                    }

                    task.GenericOwner = this;
                    task.PropertyName = taskProperty.Name;

                    m_tasks.Add(taskProperty.Name, task);
                }
                catch (Exception e)
                {
                    MyLog.ERROR.WriteLine("Automated task creation failed: " + e.Message);
                }
            }

            InitializeTaskGroups();
        }

        private void InitializeTaskGroups()
        {
            TaskGroups.Clear();

            foreach (var task in m_tasks.Values.Where(task => !string.IsNullOrEmpty(task.TaskGroupName)))
            {
                TaskGroup group;
                if (!TaskGroups.TryGetValue(task.TaskGroupName, out group))
                {
                    group = new TaskGroup(this, task.TaskGroupName);
                    TaskGroups[group.GroupName] = group;
                }

                group.Tasks.Add(task);
            }
        }

        internal void FinalizeTasksDeserialization()
        {
            //to prevent nodegroup deserialization errors
            if (m_tasks == null)
                m_tasks = new Dictionary<string, MyTask>();

            //compatibility issues (no PropertyName attributes inside brain file Tasks sections)
            Dictionary<string, PropertyInfo> taskTypeTable = new Dictionary<string, PropertyInfo>();
            foreach (PropertyInfo taskProperty in GetInfo().KnownTasks.Values)
            {
                if (!taskTypeTable.ContainsKey(taskProperty.PropertyType.FullName))
                {
                    taskTypeTable[taskProperty.PropertyType.FullName] = taskProperty;
                }
            }

            //for concurrent change purposes
            List<MyTask> tasks = new List<MyTask>(m_tasks.Values);

            //1st pass: iterate deserialized tasks and check if known (brain file may contains obsolete tasks)
            foreach (MyTask task in tasks)
            {
                if (GetInfo().KnownTasks.ContainsKey(task.PropertyName))
                {
                    task.GenericOwner = this;
                    GetInfo().KnownTasks[task.PropertyName].SetValue(this, task);
                }
                else if (task.PropertyName.StartsWith(MyTaskSerializer.NO_PROPERTY_NAME_KEY))
                {
                    string taskTypeName = task.PropertyName.Replace(MyTaskSerializer.NO_PROPERTY_NAME_KEY, "");
                    m_tasks.Remove(task.PropertyName);   

                    if (taskTypeTable.ContainsKey(taskTypeName))
                    {                                             
                        task.PropertyName = taskTypeTable[taskTypeName].Name;
                        m_tasks[task.PropertyName] = task;

                        task.GenericOwner = this;
                        taskTypeTable[taskTypeName].SetValue(this, task);
                    }                    
                }
                else 
                {
                    MyLog.WARNING.WriteLine("Unexpected task found. (" + this.GetType().Name + "->" + task.PropertyName + ")");
                    m_tasks.Remove(task.PropertyName);
                }
            }

            //2nd pass: iterate known tasks for missing tasks (new tasks missing in the old brain file)
            foreach (PropertyInfo taskProperty in GetInfo().KnownTasks.Values)
            {
                MyTask task = taskProperty.GetValue(this) as MyTask;

                if (task != null)
                {
                    if (!m_tasks.ContainsKey(taskProperty.Name))
                    {
                        m_tasks[taskProperty.Name] = task;
                    }

                    //enable all design time tasks regardless of if they were in brain file or not
                    if (task.DesignTime)
                    {
                        task.Enabled = true;
                    }                    
                }
            }

            InitializeTaskGroups();
        }

        #endregion

        #region Task groups

        public Dictionary<string, TaskGroup> TaskGroups { get; private set; }

        #endregion

        public MyExecutionBlock ExecutionBlock { get; set; }

        /// <summary>
        /// This allows the node implementations to react to the simulation's state.
        /// This is useful for e.g. resource management.
        /// </summary>
        public virtual void OnSimulationStateChanged(MySimulationHandler.StateEventArgs args)
        {
        }

        internal protected MyWorkingNode()
        {
            m_tasks = new Dictionary<string, MyTask>();
            TaskGroups = new Dictionary<string, TaskGroup>();

            var factory = this as IMyCustomTaskFactory;
            if (factory != null)
                factory.CreateTasks();

            FinalizeTaskCreation();
        }

        internal override void ValidateMandatory(MyValidator validator)
        {
            base.ValidateMandatory(validator);
            
            if (LoadOnStart || validator.Simulation.LoadAllNodesData)
            {
                if (MyMemoryBlockSerializer.TempDataExists(this))
                {
                    validator.AddInfo(this, "Node will load data from temporal storage.");
                }
                else if (DataFolder != null && DataFolder != String.Empty)
                {
                    validator.AddInfo(this, "Node will load data from user defined folder: " + DataFolder);
                }
                else if (validator.Simulation.LoadAllNodesData && ! (String.IsNullOrEmpty(validator.Simulation.GlobalDataFolder)))
                {
                    validator.AddInfo(this, "Node will load data from user defined folder: "
                        + validator.Simulation.GlobalDataFolder + "\\" + MyMemoryBlockSerializer.GetNodeFolder(this));
                }
                else if (validator.Simulation.LoadAllNodesData && (String.IsNullOrEmpty(validator.Simulation.GlobalDataFolder)))
                {
                    validator.AddInfo(this, "Node will load data from temporal storage.");
                }
                else
                {
                    validator.AddWarning(this, "LoadOnStart is active but no temporal data and no local or global data folder is set. Data will NOT be loaded.");
                }
            }

            validator.AssertInfo(!(SaveOnStop || validator.Simulation.SaveAllNodesData), this, "Node will save data to temporal storage before stop.");

            foreach (PropertyInfo pInfo in GetInfo().OwnedMemoryBlocks)
            {
                MyAbstractMemoryBlock mb = (pInfo.GetValue(this) as MyAbstractMemoryBlock);
                validator.AssertError(mb.Count >= 0, this, "Size of " + mb.Name + " memory block cannot be negative.");
            }

            List<PropertyInfo> inputBlocks = GetInfo().InputBlocks;           

            for (int i = 0; i < inputBlocks.Count; i++)
            {
                PropertyInfo pInfo = inputBlocks[i];

                if (GetAbstractInput(i) != pInfo.GetValue(this)) 
                {
                    validator.AddError(this, "Incompatible memory block for \"" + pInfo.Name + "\" (" + GetAbstractInput(i).GetType().GenericTypeArguments[0].Name + " != " + pInfo.PropertyType.GenericTypeArguments[0].Name + ")");                
                }
            }
        }
    }       
}
