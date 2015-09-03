using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;

namespace GoodAI.Core.Execution
{
    public abstract class MySimulation
    {
        public uint SimulationStep { get; protected set; }

        public bool LoadAllNodesData { get; set; }
        public bool SaveAllNodesData { get; set; }
        public int AutoSaveInterval { get; set; }   

        public string GlobalDataFolder { get; set; }        

        public bool InDebugMode { get; set; }
        public ExecutionStepMode DebugStepMode { get; set; }

        public MyExecutionBlock[] CurrentDebuggedBlocks { get; internal set; }

        public enum ExecutionStepMode
        {
            STEP_OVER,
            STEP_INTO,
            STEP_OUT
        }

        public IMyExecutionPlanner ExecutionPlanner { get; set; }
        public IMyPartitionStrategy PartitioningStrategy { get; set; }

        public MyExecutionPlan[] ExecutionPlan { get; protected set; }
        public List<MyWorkingNode>[] NodePartitioning { get; protected set; }

        protected bool m_errorOccured;
        protected Exception m_lastException;
                       
        public MySimulation()
        {
            AutoSaveInterval = 0;
            GlobalDataFolder = String.Empty;
            LoadAllNodesData = false;
        }

        public virtual void Init()
        {
            SimulationStep = 0;

            for (int i = 0; i < CurrentDebuggedBlocks.Length; i++)
            {
                CurrentDebuggedBlocks[i] = null;
            }
        }

        public abstract void AllocateMemory();
        public abstract void PerformStep(bool stepByStepRun);
        public abstract void FreeMemory();

        public virtual void Clear()
        {            
            NodePartitioning = null;
            ExecutionPlan = null;
        }

        public abstract void Finish();

        public void Schedule(MyProject project)
        {
            MyExecutionPlan singleCoreExecutionPlan = ExecutionPlanner.CreateExecutionPlan(project);
            ExecutionPlan = PartitioningStrategy.Divide(singleCoreExecutionPlan);

            //TODO: remove this and replace with proper project traversal to find nodes with no tasks!
            ExtractPartitioningFromExecutionPlan();
        }

        private void ExtractPartitioningFromExecutionPlan()
        {
            HashSet<MyWorkingNode>[] indexTable = new HashSet<MyWorkingNode>[ExecutionPlan.Length];
            NodePartitioning = new List<MyWorkingNode>[ExecutionPlan.Length];

            MyExecutionBlock.IteratorAction extractNodesAction = delegate(IMyExecutable executable)
            {
                if (executable is MyTask)
                {
                    MyWorkingNode taskOwner = (executable as MyTask).GenericOwner;
                    indexTable[taskOwner.GPU].Add(taskOwner);
                }
            };

            for (int i = 0; i < ExecutionPlan.Length; i++)
            {
                indexTable[i] = new HashSet<MyWorkingNode>();

                ExecutionPlan[i].InitStepPlan.Iterate(true, extractNodesAction);
                ExecutionPlan[i].StandardStepPlan.Iterate(true, extractNodesAction);

                NodePartitioning[i] = new List<MyWorkingNode>(indexTable[i]);
            }
        }        
    }

    public class MyLocalSimulation : MySimulation
    {
        private MyThreadPool m_threadPool;
        protected bool m_debugStepComplete; 

        public MyLocalSimulation()
        {
            m_threadPool = new MyThreadPool(MyKernelFactory.Instance.DevCount, InitCore, ExecuteCore);
            m_threadPool.StartThreads();

            try
            {
                ExecutionPlanner = new MyDefaultExecutionPlanner()
                {
                    PlanSignalTasks = true
                };

                PartitioningStrategy = new MyAllInOneGPUPartitioning(MyKernelFactory.Instance.DevCount, 0);
                CurrentDebuggedBlocks = new MyExecutionBlock[MyKernelFactory.Instance.DevCount];
            }
            catch (Exception e)
            {
                m_threadPool.Finish();
                throw e;
            }
        }

        /// <summary>
        /// Creates execution plan for project
        /// </summary>
        public override void Init()
        {
            base.Init();

            for (int i = 0; i < NodePartitioning.Length; i++)
            {
                MyKernelFactory.Instance.SetCurrent(i);

                foreach (MyWorkingNode node in NodePartitioning[i])
                {
                    //TODO: fix UI to not flicker and disable next line to clear signals after every simulation step
                    node.ClearSignals();
                    node.InitTasks();
                }
            }
        }

        public override void AllocateMemory()
        {
            for (int i = 0; i < NodePartitioning.Length; i++)
            {                
                foreach (MyWorkingNode node in NodePartitioning[i])
                {
                    MyKernelFactory.Instance.SetCurrent(i);
                    MyMemoryManager.Instance.AllocateBlocks(node, false);       
                }
            }            
        }           

        /// <summary>
        /// Performs one step of simulation
        /// </summary>
        public override void PerformStep(bool stepByStepRun)
        {
            m_debugStepComplete = true;
            m_errorOccured = false;

            m_threadPool.ResumeThreads(ExecutionPlan);

            if (m_errorOccured)
            {                
                if (m_lastException != null) 
                {                    
                    throw m_lastException;
                }   
                else 
                {
                    throw new MySimulationException(-1, "Unknown simulation exception occured");
                }
            }

            //mainly for observers
            if (InDebugMode && stepByStepRun)
            {
                for (int i = 0; i < NodePartitioning.Length; i++)
                {
                    List<MyWorkingNode> nodeList = NodePartitioning[i];
                    MyKernelFactory.Instance.SetCurrent(i);

                    for (int j = 0; j < nodeList.Count; j++)
                    {
                        MyWorkingNode node = nodeList[j];
                        MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
                    }
                }
            }

            if (!InDebugMode || m_debugStepComplete)
            {
                bool doAutoSave = SimulationStep > 0 && AutoSaveInterval > 0 && SimulationStep % AutoSaveInterval == 0;

                if (doAutoSave)
                {
                    MyLog.INFO.WriteLine("Autosave (" + SimulationStep + " steps)");
                }

                for (int i = 0; i < NodePartitioning.Length; i++)
                {
                    List<MyWorkingNode> nodeList = NodePartitioning[i];
                    MyKernelFactory.Instance.SetCurrent(i);

                    if (SimulationStep == 0)
                    {
                        LoadBlocks(nodeList);
                    }

                    if (doAutoSave)
                    {
                        SaveBlocks(nodeList);
                    }

                    for (int j = 0; j < nodeList.Count; j++)
                    {
                        MyWorkingNode node = nodeList[j];

                        //TODO: fix UI to not flicker and enable this line to clear signals after every simulation step
                        //node.ClearSignals();
                        
                        MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
                    }
                }
                SimulationStep++;
            }
        }

        private void InitCore(int coreNumber)
        {
            MyKernelFactory.Instance.SetCurrent(coreNumber);
        }

        private void ExecuteCore(int coreNumber)
        {
            try
            {
                if (InDebugMode)
                {
                    MyExecutionBlock currentBlock = CurrentDebuggedBlocks[coreNumber];

                    if (SimulationStep == 0 && currentBlock == null)
                    {
                        ExecutionPlan[coreNumber].InitStepPlan.Reset();
                        currentBlock = ExecutionPlan[coreNumber].InitStepPlan;
                        m_debugStepComplete = false;
                    }

                    MyExecutionBlock lastBlock = null;

                    do
                    {
                        lastBlock = currentBlock;
                        currentBlock.SimulationStep = SimulationStep;
                        currentBlock = currentBlock.ExecuteStep();
                    }
                    while (currentBlock != null && currentBlock.CurrentChild == null);

                    if (currentBlock == null)
                    {
                        ExecutionPlan[coreNumber].StandardStepPlan.Reset();
                        currentBlock = ExecutionPlan[coreNumber].StandardStepPlan;
                    }
                    else
                    {
                        m_debugStepComplete = false;
                    }

                    CurrentDebuggedBlocks[coreNumber] = currentBlock;
                }
                else //not in debug mode
                {
                    if (SimulationStep == 0)
                    {
                        ExecutionPlan[coreNumber].InitStepPlan.SimulationStep = 0;
                        ExecutionPlan[coreNumber].InitStepPlan.Execute();
                    }

                    //TODO: here should be else! (but some module are not prepared for this)
                    ExecutionPlan[coreNumber].StandardStepPlan.SimulationStep = SimulationStep;
                    ExecutionPlan[coreNumber].StandardStepPlan.Execute();
                }
            }
            catch (Exception e)
            {
                m_errorOccured = true;
                m_lastException = new MySimulationException(coreNumber, e.Message, e);

                MyKernelFactory.Instance.MarkContextDead(coreNumber);               
            }
        }

        public override void FreeMemory()
        {
            for (int i = 0; i < NodePartitioning.Length; i++)
            {
                MyKernelFactory.Instance.SetCurrent(i);

                SaveBlocks(NodePartitioning[i]);

                foreach (MyWorkingNode node in NodePartitioning[i])
                {                    
                    MyMemoryManager.Instance.FreeBlocks(node, false);
                }
            }
        }

        public override void Finish()
        {
            m_threadPool.FinishFromSTAThread();
        }

        private void LoadBlocks(List<MyWorkingNode> nodeList)
        {            
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();                    

            for (int j = 0; j < nodeList.Count; j++)
            {
                MyWorkingNode node = nodeList[j];

                if (LoadAllNodesData || node.LoadOnStart)
                {
                    foreach (MyAbstractMemoryBlock mb in MyMemoryManager.Instance.GetBlocks(node))
                    {
                        if (mb.Persistable)
                        {
                            serializer.LoadBlock(mb, GlobalDataFolder);
                        }
                    }
                }
            }
        }

        private void SaveBlocks(List<MyWorkingNode> nodeList)
        {            
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();

            for (int j = 0; j < nodeList.Count; j++)
            {
                MyWorkingNode node = nodeList[j];

                if (SaveAllNodesData || node.SaveOnStop)
                {
                    foreach (MyAbstractMemoryBlock mb in MyMemoryManager.Instance.GetBlocks(node))
                    {
                        if (mb.Persistable)
                        {
                            serializer.SaveBlock(mb);
                        }
                    }
                }
            }
        }
    }

    public class MySimulationException : Exception 
    {
        public int CoreNumber { get; private set; }

        public MySimulationException(int coreNumber, string message) : base(message)
        {
            CoreNumber = coreNumber;
        }

        public MySimulationException(int coreNumber, string message, Exception innerException)
            : base(message, innerException)
        {
            CoreNumber = coreNumber;
        }
    }
 }
