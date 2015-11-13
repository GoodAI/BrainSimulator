using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using GoodAI.Platform.Core.Utils;

namespace GoodAI.Core.Execution
{
    public enum DebugStepMode
    {
        None,
        StepInto,
        StepOver,
        StepOut
    }

    public abstract class MySimulation
    {

        public uint SimulationStep { get; protected set; }

        public bool LoadAllNodesData { get; set; }
        public bool SaveAllNodesData { get; set; }
        public int AutoSaveInterval { get; set; }

        public bool IsFinished { get; protected set; }

        public string GlobalDataFolder { get; set; }

        public bool InDebugMode { get; set; }
        // Specifies the block that the simulation should stop at.
        public readonly ISet<IMyExecutable> Breakpoints = new HashSet<IMyExecutable>();
        protected DebugStepMode DebugStepMode = DebugStepMode.None;
        protected MyExecutionBlock StopWhenTouchedBlock;

        public delegate void DebugTargetEncounteredHandler(object sender, EventArgs args);
        public event DebugTargetEncounteredHandler DebugTargetReached;

        protected void EmitDebugTargetReached()
        {
            DebugStepMode = DebugStepMode.None;
            if (DebugTargetReached != null)
                DebugTargetReached(this, EventArgs.Empty);
        }

        public MyExecutionBlock[] CurrentDebuggedBlocks { get; internal set; }

        public IMyExecutionPlanner ExecutionPlanner { get; set; }
        public IMyPartitionStrategy PartitioningStrategy { get; set; }

        public MyExecutionPlan[] ExecutionPlan { get; protected set; }
        public List<MyWorkingNode>[] NodePartitioning { get; protected set; }


        protected bool m_errorOccured;
        protected Exception m_lastException;

        public void OnStateChanged(object sender, MySimulationHandler.StateEventArgs args)
        {
            foreach (MyWorkingNode node in NodePartitioning.SelectMany(nodeList => nodeList))
                node.OnSimulationStateChanged(args);
        }

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

            IsFinished = false;
        }

        public abstract void AllocateMemory();
        public abstract void PerformStep(bool stepByStepRun);
        public abstract void FreeMemory();

        public abstract void StepOver();
        public abstract void StepInto();
        public abstract void StepOut();

        public virtual void Clear()
        {
            NodePartitioning = null;
            ExecutionPlan = null;
        }

        public void Finish()
        {
            DoFinish();
            Clear();
            IsFinished = true;
        }

        protected abstract void DoFinish();

        public void Schedule(MyProject project)
        {
            MyExecutionPlan singleCoreExecutionPlan = ExecutionPlanner.CreateExecutionPlan(project);
            ExecutionPlan = PartitioningStrategy.Divide(singleCoreExecutionPlan);

            //TODO: remove this and replace with proper project traversal to find nodes with no tasks!
            ExtractPartitioningFromExecutionPlan();
        }

        private void ExtractPartitioningFromExecutionPlan()
        {
            if (ExecutionPlan == null)
                throw new SimulationControlException("The simulation is not set up.");

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

            ExecutionPlan.EachWithIndex((item, i) =>
            {
                indexTable[i] = new HashSet<MyWorkingNode>();

                ExecutionPlan[i].InitStepPlan.Iterate(true, extractNodesAction);
                ExecutionPlan[i].StandardStepPlan.Iterate(true, extractNodesAction);

                NodePartitioning[i] = new List<MyWorkingNode>(indexTable[i]);
            });
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

            if (NodePartitioning == null)
                throw new SimulationControlException("The execution plan is not set up.");

            NodePartitioning.EachWithIndex((partition, i) =>
            {
                MyKernelFactory.Instance.SetCurrent(i);

                foreach (MyWorkingNode node in partition)
                {
                    //TODO: fix UI to not flicker and disable next line to clear signals after every simulation step
                    node.ClearSignals();
                    node.InitTasks();
                }
            });
        }

        public override void AllocateMemory()
        {
            if (NodePartitioning == null)
                throw new SimulationControlException("The execution plan is not set up.");

            NodePartitioning.EachWithIndex((partition, i) =>
            {
                foreach (MyWorkingNode node in partition)
                {
                    MyKernelFactory.Instance.SetCurrent(i);
                    MyMemoryManager.Instance.AllocateBlocks(node, false);
                }
            });
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
                if (NodePartitioning == null)
                    throw new SimulationControlException("The execution plan is not set up.");

                for (int i = 0; i < NodePartitioning.Length; i++)
                {
                    List<MyWorkingNode> nodeList = NodePartitioning[i];
                    MyKernelFactory.Instance.SetCurrent(i);

                    foreach (MyWorkingNode node in nodeList)
                    {
                        MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
                    }
                }
            }

            if (!InDebugMode || m_debugStepComplete)
            {
                if (NodePartitioning == null)
                    throw new SimulationControlException("The simulation is not set up.");

                bool doAutoSave = SimulationStep > 0 && AutoSaveInterval > 0 && SimulationStep % AutoSaveInterval == 0;

                if (doAutoSave)
                {
                    MyLog.INFO.WriteLine("Autosave (" + SimulationStep + " steps)");
                }

                if (NodePartitioning == null)
                    throw new SimulationControlException("The simulation is not set up.");

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

                    foreach (MyWorkingNode node in nodeList)
                    {
                        //TODO: fix UI to not flicker and enable this line to clear signals after every simulation step
                        //node.ClearSignals();

                        MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
                    }
                };
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

                    bool leavingTargetBlock = false;

                    do
                    {
                        currentBlock.SimulationStep = SimulationStep;
                        currentBlock = currentBlock.ExecuteStep();
                        if (StopWhenTouchedBlock != null && currentBlock == StopWhenTouchedBlock)
                            leavingTargetBlock = true;
                    }
                    while (currentBlock != null && currentBlock.CurrentChild == null);

                    if (currentBlock == null)
                    {
                        ExecutionPlan[coreNumber].StandardStepPlan.Reset();
                        currentBlock = ExecutionPlan[coreNumber].StandardStepPlan;
                        leavingTargetBlock = true;
                    }
                    else
                    {
                        m_debugStepComplete = false;
                    }

                    CurrentDebuggedBlocks[coreNumber] = currentBlock;

                    if (DebugStepMode != DebugStepMode.None)
                    {
                        // A step into/over/out is performed.
                        if (leavingTargetBlock)
                            // The target block is being left or the sim step is over - step over/out is finished.
                            EmitDebugTargetReached();

                        if (DebugStepMode == DebugStepMode.StepInto)
                        {
                            // Step into == one step of the simulation.
                            EmitDebugTargetReached();
                        }
                    }

                    if (Breakpoints.Contains(currentBlock.CurrentChild))
                        // A breakpoint has been reached.
                        EmitDebugTargetReached();
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

                if (e is MySimulationException)
                {
                    m_lastException = e;
                }
                else
                {
                    m_lastException = new MySimulationException(coreNumber, e.Message, e);
                    MyKernelFactory.Instance.MarkContextDead(coreNumber);
                }
            }
        }

        public override void FreeMemory()
        {
            if (NodePartitioning == null)
                return;

            NodePartitioning.EachWithIndex((partition, i) =>
            {
                MyKernelFactory.Instance.SetCurrent(i);

                SaveBlocks(partition);

                foreach (MyWorkingNode node in partition)
                {
                    MyMemoryManager.Instance.FreeBlocks(node, false);

                    node.Cleanup();
                }
            });
        }

        public override void StepOver()
        {
            // HonzaS: Using 0 because the plan will be unified with StarPU anyway.
            MyExecutionBlock currentBlock = CurrentDebuggedBlocks[0];
            if (currentBlock != null)
            {
                // If the current child is a block, pause the simulation when it's next sibling is requested.
                // That happens by visiting this node again.
                var currentChildBlock = currentBlock.CurrentChild as MyExecutionBlock;
                if (currentChildBlock != null)
                {
                    StopWhenTouchedBlock = currentBlock;
                    DebugStepMode = DebugStepMode.StepOver;
                    return;
                }
            }

            // If the current child is not a block, run for one step.
            // The sim will execute the task and move onto the next.
            DebugStepMode = DebugStepMode.StepInto;
        }

        public override void StepOut()
        {
            MyExecutionBlock currentBlock = CurrentDebuggedBlocks[0];
            if (currentBlock != null)
            {
                // This is equivalent to calling StepOver on this node's parent node.
                StopWhenTouchedBlock = currentBlock.Parent;

                // Set this so that the loops knows it should stop when it hits the end of the plan.
                DebugStepMode = DebugStepMode.StepOut;
            }
        }

        public override void StepInto()
        {
            DebugStepMode = DebugStepMode.StepInto;
        }

        private MyExecutionBlock GetNextExecutable(MyExecutionBlock executionBlock)
        {
            if (executionBlock.NextChild != null)
                return executionBlock;

            if (executionBlock.Parent != null)
                return GetNextExecutable(executionBlock.Parent);

            // This is the root and it doesn't have a "next" node.
            return null;
        }

        protected override void DoFinish()
        {
            m_threadPool.FinishFromSTAThread();
        }

        private void LoadBlocks(List<MyWorkingNode> nodeList)
        {
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();

            foreach (MyWorkingNode node in nodeList)
            {
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

            foreach (MyWorkingNode node in nodeList)
            {
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

    public class SimulationControlException : Exception
    {
        public SimulationControlException(string message) : base(message) { }
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
