using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using GoodAI.Platform.Core.Nodes;
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

        public MyExecutionPlan ExecutionPlan { get; protected set; }
        public HashSet<MyWorkingNode> AllNodes { get; protected set; }

        protected IList<IModelChanger> ModelChangingNodeGroups { get; set; }


        protected bool m_errorOccured;
        protected Exception m_lastException;
        protected MyProject m_project;

        public void OnStateChanged(object sender, MySimulationHandler.StateEventArgs args)
        {
            foreach (MyWorkingNode node in AllNodes)
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
            ResetSimulationStep();

            for (int i = 0; i < CurrentDebuggedBlocks.Length; i++)
            {
                CurrentDebuggedBlocks[i] = null;
            }

            IsFinished = false;
        }

        public void ResetSimulationStep()
        {
            SimulationStep = 0;
        }

        public abstract void AllocateMemory();
        public abstract void PerformStep(bool stepByStepRun);

        /// <summary>
        /// Indicates that the simulation is in between two simulation steps.
        /// This should be true after each PerformStep run during normal simulation, and can be false during debug.
        /// </summary>
        public abstract bool IsStepFinished { get; }
        public abstract void FreeMemory();

        public abstract void StepOver();
        public abstract void StepInto();
        public abstract void StepOut();

        public virtual void Clear()
        {
            AllNodes = null;
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
            m_project = project;
            ExecutionPlan = ExecutionPlanner.CreateExecutionPlan(project);
            //ExecutionPlan = PartitioningStrategy.Divide(singleCoreExecutionPlan);

            //TODO: remove this and replace with proper project traversal to find nodes with no tasks!
            ExtractAllNodes(m_project);
        }

        private void ExtractAllNodes(MyProject project)
        {
            AllNodes = new HashSet<MyWorkingNode>();
            ModelChangingNodeGroups = new List<IModelChanger>();

            AllNodes.Add(project.World);

            project.Network.Iterate(true, true, node =>
            {
                var workingNode = node as MyWorkingNode;
                if (workingNode != null)
                    AllNodes.Add(workingNode);

                var modelChanger = node as IModelChanger;
                if (modelChanger != null)
                    ModelChangingNodeGroups.Add(modelChanger);
            });
        }

        public void CleanTemporaryBlockData()
        {
            CleanBreakpoints();
            CleanProfilingTimes();
        }

        private void CleanBreakpoints()
        {
            var orphanedExecutables = new HashSet<IMyExecutable>(Breakpoints);
            ExecutionPlan.StandardStepPlan.Iterate(true, executable => orphanedExecutables.Remove(executable));

            foreach (var executable in orphanedExecutables)
                Breakpoints.Remove(executable);
        }

        private void CleanProfilingTimes()
        {
            CleanExecutionBlockProfilingTimes(ExecutionPlan.InitStepPlan);
            CleanExecutionBlockProfilingTimes(ExecutionPlan.StandardStepPlan);
        }

        private void CleanExecutionBlockProfilingTimes(MyExecutionBlock plan)
        {
            plan.Iterate(true, executable =>
            {
                var executableBlock = executable as MyExecutionBlock;
                if (executableBlock != null)
                    executableBlock.CleanProfilingTimes();
            });
        }

        public abstract void PerformModelChanges();
        public abstract void Reallocate();
    }

    public sealed class MyLocalSimulation : MySimulation
    {
        private readonly MyThreadPool m_threadPool;
        protected bool m_debugStepComplete;
        private bool m_debugInitInProgress;

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

            if (AllNodes == null)
                throw new SimulationControlException("The execution plan is not set up.");

            foreach (MyWorkingNode node in AllNodes)
            {
                MyKernelFactory.Instance.SetCurrent(0);

                //TODO: fix UI to not flicker and disable next line to clear signals after every simulation step
                node.ClearSignals();
                node.InitTasks();
            }
        }

        public override void AllocateMemory()
        {
            if (AllNodes == null)
                throw new SimulationControlException("The execution plan is not set up.");

            AllocateMemory(AllNodes);
        }

        private static void AllocateMemory(IEnumerable<MyWorkingNode> nodes)
        {
            foreach (MyWorkingNode node in nodes)
            {
                MyKernelFactory.Instance.SetCurrent(0);
                MyMemoryManager.Instance.AllocateBlocks(node, false);
            }
        }

        private static void AllocateMemory(MyNode node)
        {
            var workingNode = node as MyWorkingNode;
            if (workingNode == null)
                return;

            MyKernelFactory.Instance.SetCurrent(0);
            MyMemoryManager.Instance.AllocateBlocks(node, false);
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

                throw new MySimulationException(-1, "Unknown simulation exception occured");
            }

            //mainly for observers
            if (InDebugMode && stepByStepRun)
            {
                if (AllNodes == null)
                    throw new SimulationControlException("The execution plan is not set up.");

                MyKernelFactory.Instance.SetCurrent(0);

                foreach (MyWorkingNode node in AllNodes)
                    MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
            }

            if (!InDebugMode || m_debugStepComplete)
            {
                if (AllNodes == null)
                    throw new SimulationControlException("The simulation is not set up.");

                bool doAutoSave = SimulationStep > 0 && AutoSaveInterval > 0 && SimulationStep % AutoSaveInterval == 0;

                if (doAutoSave)
                {
                    MyLog.INFO.WriteLine("Autosave (" + SimulationStep + " steps)");
                }

                if (AllNodes == null)
                    throw new SimulationControlException("The simulation is not set up.");

                MyKernelFactory.Instance.SetCurrent(0);

                if (SimulationStep == 0)
                {
                    LoadBlocks(AllNodes);
                }

                if (doAutoSave)
                {
                    SaveBlocks(AllNodes);
                }

                foreach (MyWorkingNode node in AllNodes)
                {
                    //TODO: fix UI to not flicker and enable this line to clear signals after every simulation step
                    //node.ClearSignals();

                    MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
                }

                SimulationStep++;
            }
        }

        public override bool IsStepFinished { get { return m_debugStepComplete; } }

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

                    // This is the first debug step.
                    if (SimulationStep == 0 && currentBlock == null)
                    {
                        m_debugInitInProgress = true;
                        ExecutionPlan.InitStepPlan.Reset();
                        currentBlock = ExecutionPlan.InitStepPlan;
                        m_debugStepComplete = false;
                    }

                    // This checks if breakpoint was encountered, also used for "stepping".
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
                        // The current plan finished, the standard plan has to be reset and executed.
                        if (m_debugInitInProgress)
                            m_debugStepComplete = false;  // This means the init plan got finished, not the standard plan.

                        m_debugInitInProgress = false;

                        ExecutionPlan.StandardStepPlan.Reset();
                        currentBlock = ExecutionPlan.StandardStepPlan;
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
                        ExecutionPlan.InitStepPlan.SimulationStep = 0;
                        ExecutionPlan.InitStepPlan.Execute();
                    }

                    //TODO: here should be else! (but some module are not prepared for this)
                    ExecutionPlan.StandardStepPlan.SimulationStep = SimulationStep;
                    ExecutionPlan.StandardStepPlan.Execute();
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
            if (AllNodes == null)
                return;

            MyKernelFactory.Instance.SetCurrent(0);

            SaveBlocks(AllNodes);

            FreeMemory(AllNodes);
        }

        private static void FreeMemory(IEnumerable<MyWorkingNode> nodes)
        {
            foreach (MyWorkingNode node in nodes)
                FreeMemory(node);
        }

        private static void FreeMemory(MyNode node)
        {
            var workingNode = node as MyWorkingNode;
            if (workingNode == null)
                return;

            MyMemoryManager.Instance.FreeBlocks(workingNode, false);
            workingNode.Cleanup();
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

        public override void PerformModelChanges()
        {
            //var newModelChangingGroups = new List<IModelChanger>();
            // Go through the topologically ordered model changing groups and allow them to restructure.
            bool modelChanged = false;
            ModelChangingNodeGroups.EachWithIndex((changer, i) =>
            {
                var removedNodes = new List<MyWorkingNode>();
                var addedNodes = new List<MyWorkingNode>();

                if (!changer.IsModelChanging)
                    return;

                modelChanged = true;
                changer.ChangeModel(ref removedNodes, ref addedNodes);

                // Clean up memory.
                IterateNodes(removedNodes, FreeMemory);

                // Update memory blocks of the newly added nodes.
                // TODO(HonzaS): "shakedown" (the convergence of Count updates).
                foreach (MyWorkingNode node in addedNodes)
                    node.UpdateMemoryBlocks();

                // Allocate new memory.
                IterateNodes(addedNodes, AllocateMemory);

                //newModelChangingGroups.Add(changer);
                //IterateNodes(addedNodes, node =>
                //{
                //    var newChanger = node as IModelChanger;
                //    if (newChanger != null)
                //        newModelChangingGroups.Add(newChanger);
                //});
            });

            //ModelChangingNodeGroups = newModelChangingGroups;

            // Refresh topological ordering.
            if (!modelChanged)
                return;

            MySimulationHandler.OrderNetworkNodes(m_project.Network);

            Schedule(m_project);
        }

        private static void IterateNodes(IEnumerable<MyWorkingNode> nodes, MyNodeGroup.IteratorAction action)
        {
            foreach (MyWorkingNode node in nodes)
            {
                action(node);

                var group = node as MyNodeGroup;
                if (group != null)
                    group.Iterate(true, action);
            }
        }

        public override void Reallocate()
        {
        }

        private void LoadBlocks(IEnumerable<MyWorkingNode> nodeList)
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

        private void SaveBlocks(IEnumerable<MyWorkingNode> nodeList)
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
