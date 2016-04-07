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
using GoodAI.TypeMapping;

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
        protected static int MAX_BLOCKS_UPDATE_ATTEMPTS = 20;

        public uint SimulationStep { get; protected set; }

        public bool LoadAllNodesData
        {
            get { return (m_project != null) ? m_project.LoadAllNodesData : false; }
        }

        public bool SaveAllNodesData
        {
            get { return (m_project != null) ? m_project.SaveAllNodesData : false; }
        }

        public int AutoSaveInterval { get; set; }

        public bool IsFinished { get; protected set; }

        public string GlobalDataFolder { get; set; }

        public bool InDebugMode { get; set; }
        // Specifies the block that the simulation should stop at.
        public readonly ISet<IMyExecutable> Breakpoints = new HashSet<IMyExecutable>();
        protected DebugStepMode DebugStepMode = DebugStepMode.None;
        protected MyExecutionBlock StopWhenTouchedBlock;

        public MyValidator Validator { get; private set; }

        public class ModelChangedEventArgs : EventArgs
        {
            public MyNode Node { get; set; }
        }

        public event EventHandler<ModelChangedEventArgs> ModelChanged;

        public event EventHandler DebugTargetReached;

        protected void EmitModelChanged(MyNode node)
        {
            if (ModelChanged != null)
                ModelChanged(this, new ModelChangedEventArgs {Node = node});
        }

        protected void EmitDebugTargetReached()
        {
            DebugStepMode = DebugStepMode.None;
            if (DebugTargetReached != null)
                DebugTargetReached(this, EventArgs.Empty);
        }

        public abstract bool UpdateMemoryModel(MyProject project, List<MyNode> orderedNodes);

        public MyExecutionBlock CurrentDebuggedBlock { get; internal set; }

        public IMyExecutionPlanner ExecutionPlanner { get; set; }

        public MyExecutionPlan ExecutionPlan { get; protected set; }
        public HashSet<MyWorkingNode> AllNodes { get; protected set; }

        protected IList<IModelChanger> ModelChangingNodes { get; set; }


        protected bool m_errorOccured;
        protected Exception m_lastException;
        protected MyProject m_project;

        protected bool m_memoryLoadingDone;

        public void OnStateChanged(object sender, MySimulationHandler.StateEventArgs args)
        {
            foreach (MyWorkingNode node in AllNodes)
                node.OnSimulationStateChanged(args);
        }

        public MySimulation(MyValidator validator)
        {
            AutoSaveInterval = 0;
            GlobalDataFolder = String.Empty;

            Validator = validator;
            validator.Simulation = this;
        }

        public virtual void Init()
        {
            ResetSimulationStep();

            CurrentDebuggedBlock = null;

            IsFinished = false;

            m_memoryLoadingDone = false;
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
        public bool IsStepFinished { get; protected internal set; }

        public abstract bool IsChangingModel { get; }
        public abstract void FreeMemory(bool didCrash);

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

        public void Schedule(MyProject project, IEnumerable<MyWorkingNode> newNodes = null)
        {
            // If there are any init tasks in the current plan, copy them over to the new one.
            // This is mostly for the first simulation step if there is also a model change.
            MyExecutionBlock oldPlan = null;
            if (ExecutionPlan != null)
                oldPlan = ExecutionPlan.InitStepPlan;

            m_project = project;
            ExecutionPlan = ExecutionPlanner.CreateExecutionPlan(project, newNodes);

            if (oldPlan != null)
            {
                var newInitPlan = new List<IMyExecutable>();
                newInitPlan.AddRange(oldPlan.Children);
                newInitPlan.AddRange(ExecutionPlan.InitStepPlan.Children);
                ExecutionPlan.InitStepPlan = new MyExecutionBlock(newInitPlan.ToArray()) {Name = oldPlan.Name};
            }

            ExtractAllNodes(m_project);

            // Allow subclasses to react to re-scheduling.
            ScheduleChanged();
        }

        protected virtual void ScheduleChanged() {}

        private void ExtractAllNodes(MyProject project)
        {
            AllNodes = new HashSet<MyWorkingNode>();
            ModelChangingNodes = new List<IModelChanger>();

            AllNodes.Add(project.World);

            var worldChanger = project.World as IModelChanger;
            if (worldChanger != null)
                ModelChangingNodes.Add(worldChanger);

            project.Network.Iterate(true, true, node =>
            {
                var workingNode = node as MyWorkingNode;
                if (workingNode != null)
                    AllNodes.Add(workingNode);

                var modelChanger = node as IModelChanger;
                if (modelChanger != null)
                    ModelChangingNodes.Add(modelChanger);
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
            if (ExecutionPlan.InitStepPlan != null)
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

        public void Validate(MyProject project = null)
        {
            Validator.ClearValidation();

            if (project == null)
                project = m_project;

            project.World.ValidateWorld(Validator);                
            project.Network.Validate(Validator);
        }
    }

    public sealed class MyLocalSimulation : MySimulation
    {
        enum ExecutionPhase
        {
            Initialization,
            PreStandard,
            Standard
        }

        private readonly MyThreadPool m_threadPool;
        private bool m_stepComplete = true;
        private ExecutionPhase m_executionPhase;
        private bool m_isChangingModel;

        public MyLocalSimulation(MyValidator validator, IMyExecutionPlanner executionPlanner) : base(validator)
        {
            m_threadPool = new MyThreadPool(MyKernelFactory.Instance.DevCount, InitCore, ExecutePlan);
            m_threadPool.StartThreads();

            try
            {
                ExecutionPlanner = executionPlanner;
            }
            catch (Exception e)
            {
                m_threadPool.Finish();
                throw e;
            }
        }

        public override void Init()
        {
            base.Init();

            CheckSimulationSetup();

            foreach (MyWorkingNode node in AllNodes)
            {
                MyKernelFactory.Instance.SetCurrent(0);

                //TODO: fix UI to not flicker and disable next line to clear signals after every simulation step
                node.ClearSignals();
                node.InitTasks();
            }

            IsStepFinished = true;
        }

        public override void AllocateMemory()
        {
            CheckSimulationSetup();

            AllocateMemory(AllNodes);
        }

        private static void AllocateMemory(IEnumerable<MyWorkingNode> nodes)
        {
            MyKernelFactory.Instance.SetCurrent(0);
            foreach (MyWorkingNode node in nodes)
                MyMemoryManager.Instance.AllocateBlocks(node, false);
        }

        /// <summary>
        /// Performs one step of simulation
        /// </summary>
        public override void PerformStep(bool stepByStepRun)
        {
            // Debug step is presumed finished, but can change to false during the execution.
            m_stepComplete = true;
            IsStepFinished = false;
            m_errorOccured = false;

            if (m_executionPhase == ExecutionPhase.Initialization || m_executionPhase == ExecutionPhase.Standard)
                ResumeThreads();

            // The phase might have changed during ResumeThreads. If it did, the loading of blocks must happen now.
            if (m_executionPhase == ExecutionPhase.PreStandard)
            {
                // TODO(HonzaS): When we enable block loading during simulation, change this to support it.
                // There should be a field (m_blockLoadingNodes) that gets processed here.
                if (SimulationStep == 0)
                {
                    MyKernelFactory.Instance.SetCurrent(0);
                    LoadBlocks(AllNodes);
                    m_memoryLoadingDone = true;
                }

                m_executionPhase = ExecutionPhase.Standard;

                // In normal mode, we want to run the whole step - we don't wait for the next PerformStep call.
                if (!InDebugMode)
                    ResumeThreads();
            }

            //mainly for observers
            if (InDebugMode && stepByStepRun)
            {
                CheckSimulationSetup();

                MyKernelFactory.Instance.SetCurrent(0);

                foreach (MyWorkingNode node in AllNodes)
                    MyMemoryManager.Instance.SynchronizeSharedBlocks(node, false);
            }

            if (m_stepComplete)
            {
                CheckSimulationSetup();

                bool doAutoSave = SimulationStep > 0 && AutoSaveInterval > 0 && SimulationStep%AutoSaveInterval == 0;

                if (doAutoSave)
                {
                    MyLog.INFO.WriteLine("Autosave (" + SimulationStep + " steps)");
                }

                MyKernelFactory.Instance.SetCurrent(0);

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

        private void ResumeThreads()
        {
            m_threadPool.ResumeThreads();

            if (m_errorOccured)
            {
                if (m_lastException != null)
                    throw m_lastException;

                throw new MySimulationException("Unknown simulation exception occured");
            }
        }

        private void CheckSimulationSetup()
        {
            if (AllNodes == null)
                throw new SimulationControlException("The execution plan is not set up.");
        }

        public override bool IsChangingModel { get { return m_isChangingModel; } }

        private void InitCore(int coreNumber)
        {
            MyKernelFactory.Instance.SetCurrent(coreNumber);
        }

        protected override void ScheduleChanged()
        {
            m_executionPhase = ExecutionPhase.Initialization;

            CurrentDebuggedBlock = ExecutionPlan.InitStepPlan;
        }

        // TODO(HonzaS): The coreNumber parameter is not needed, remove it with the StarPU merge.
        private void ExecutePlan(int coreNumber)
        {
            try
            {
                if (InDebugMode)
                {
                    ExecuteDebugStep();
                }
                else
                {
                    ExecuteFullStep();
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
                    m_lastException = new MySimulationException(e.Message, e);
                    MyKernelFactory.Instance.MarkContextDead(0);
                }
            }
            finally
            {
                IsStepFinished = m_stepComplete;
            }
        }

        private void ExecuteFullStep()
        {
            if (m_executionPhase == ExecutionPhase.Initialization)
            {
                if (ExecutionPlan.InitStepPlan != null)
                {
                    ExecutionPlan.InitStepPlan.SimulationStep = SimulationStep;
                    ExecutionPlan.InitStepPlan.Execute();
                }
                m_executionPhase = ExecutionPhase.PreStandard;
            }
            else if (m_executionPhase == ExecutionPhase.Standard)
            {
                ExecutionPlan.StandardStepPlan.SimulationStep = SimulationStep;
                ExecutionPlan.StandardStepPlan.Execute();
                ExecutionPlan.InitStepPlan = null;

                // We don't go to Initialization, because init currently only reappears when rescheduling is done,
                // which sets this accordingly.
                m_executionPhase = ExecutionPhase.PreStandard;
            }
        }

        private void ExecuteDebugStep()
        {
            MyExecutionBlock currentBlock = CurrentDebuggedBlock;

            if (currentBlock == null)
            {
                if (m_executionPhase == ExecutionPhase.Initialization)
                {
                    // The next step should be the beginning of the initialization plan.
                    if (ExecutionPlan.InitStepPlan != null)
                    {
                        // There is an initialization plan, take the first step.
                        ExecutionPlan.InitStepPlan.Reset();
                        currentBlock = ExecutionPlan.InitStepPlan;
                    }
                    else
                    {
                        // There is no initialization plan, go to PreStandard and stop because the loading of
                        // block data might be needed.
                        m_executionPhase = ExecutionPhase.PreStandard;
                        return;
                    }
                }
                else if (m_executionPhase == ExecutionPhase.Standard)
                {
                    ExecutionPlan.StandardStepPlan.Reset();
                    currentBlock = ExecutionPlan.StandardStepPlan;
                }
            }

            // This checks if breakpoint was encountered, also used for "stepping".
            bool leavingTargetBlock = false;

            do
            {
                currentBlock.SimulationStep = SimulationStep;
                currentBlock = currentBlock.ExecuteStep();
                if (StopWhenTouchedBlock != null && currentBlock == StopWhenTouchedBlock)
                    leavingTargetBlock = true;
            } while (currentBlock != null && currentBlock.CurrentChild == null);

            if (currentBlock == null)
            {
                // The current plan finished.

                if (m_executionPhase == ExecutionPhase.Initialization)
                {
                    // This means the init plan got finished, not the standard plan.
                    m_stepComplete = false;
                }
                else
                {
                    // This means the standard plan finished, remove the init plan (debug window will reset).
                    ExecutionPlan.InitStepPlan = null;
                }

                // If rescheduling happens, this will be set to "Initialization" again, if not, we need to 
                // perform just the standard plan again.
                m_executionPhase = ExecutionPhase.PreStandard;
                leavingTargetBlock = true;
            }
            else
            {
                m_stepComplete = false;
            }

            CurrentDebuggedBlock = currentBlock;

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

            if (currentBlock != null && Breakpoints.Contains(currentBlock.CurrentChild))
                // A breakpoint has been reached.
                EmitDebugTargetReached();
        }

        public override void FreeMemory(bool didCrash)
        {
            if (AllNodes == null)
                return;

            MyKernelFactory.Instance.SetCurrent(0);

            if (!didCrash || m_memoryLoadingDone)  // Don't save data when we crashed before even attempting to load them.
                SaveBlocks(AllNodes);
            else
                MyLog.WARNING.WriteLine("State saving skipped becase of crash before loading.");

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
            MyExecutionBlock currentBlock = CurrentDebuggedBlock;
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
            MyExecutionBlock currentBlock = CurrentDebuggedBlock;
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

        protected override void DoFinish()
        {
            m_threadPool.FinishFromSTAThread();
        }

        public override void Clear()
        {
            base.Clear();
            m_stepComplete = true;
        }

        /// <summary>
        /// Go through the topologically ordered model changing groups and allow them to restructure.
        /// </summary>
        public override void PerformModelChanges()
        {
            m_isChangingModel = true;
            var modelChanges = TypeMap.GetInstance<IModelChanges>();
            var changersActivated = new List<MyNode>();

            bool modelChanged = false;
            foreach (IModelChanger changer in ModelChangingNodes)
            {
                bool nodeChanged = changer.ChangeModel(modelChanges);
                if (nodeChanged)
                    changersActivated.Add(changer.AffectedNode);

                modelChanged |= nodeChanged;
            }

            if (!modelChanged)
                return;

            SetupAfterModelChange(modelChanges, changersActivated);
            m_isChangingModel = false;
        }

        private void SetupAfterModelChange(IModelChanges modelChanges, List<MyNode> changersActivated)
        {
            // Clean up memory.
            IterateNodes(modelChanges.RemovedNodes, DestroyNode);

            // This must happen before UpdateMemoryModel() because some nodes touch kernels in UpdateMemoryBlocks().
            MyKernelFactory.Instance.SetCurrent(0);

            // Refresh topological ordering.
            List<MyNode> orderedNodes = MySimulationHandler.OrderNetworkNodes(m_project.Network);

            // Update the whole memory model.
            // TODO(HonzaS): This may break things, check.
            // We'll need to forbid changing of count after the simulation has started with the exception of added nodes.
            // However, the added nodes may lead to reallocation of blocks - deal with it.
            bool updatesNotConverged = UpdateMemoryModel(m_project, orderedNodes);

            Validator.ClearValidation();

            // Validate new nodes.
            IterateNodes(modelChanges.AddedNodes, ValidateNode);

            Validator.AssertError(!updatesNotConverged, m_project.Network, "Possible infinite loop in memory block sizes.");

            if (!Validator.ValidationSucessfull)
                throw new InvalidOperationException("Validation failed for the changed model.");

            // Reschedule.
            Schedule(m_project, modelChanges.AddedNodes);

            // Init nodes
            IterateNodes(modelChanges.AddedNodes, InitNode);

            // Allocate memory
            IEnumerable<MyWorkingNode> nodesToAllocate =
                modelChanges.AddedNodes.Where(node => MyMemoryManager.Instance.IsRegistered(node));
            IterateNodes(nodesToAllocate, AllocateNode);

            foreach (MyNode node in changersActivated)
                EmitModelChanged(node);
        }

        private static void InitNode(MyNode node)
        {
            var workingNode = node as MyWorkingNode;
            if (workingNode == null)
                return;

            workingNode.ClearSignals();
            workingNode.InitTasks();
        }

        private static void AllocateNode(MyNode node)
        {
            // TODO(HonzaS): Does the allocation need to be done in a separate loop?
            MyMemoryManager.Instance.AllocateBlocks(node, false);
        }

        private void ValidateNode(MyNode node)
        {
            node.ValidateMandatory(Validator);
            node.Validate(Validator);
        }

        private static void DestroyNode(MyNode node)
        {
            FreeMemory(node);
            node.Dispose();
        }

        public override void Reallocate()
        {
            // This will allocate memory on the device. The CUDA context needs to be set up.
            MyKernelFactory.Instance.SetCurrent(0);

            // TODO(HonzaS): cache the ordered nodes if they have been ordered in model changes.
            foreach (MyNode node in MySimulationHandler.OrderNetworkNodes(m_project.Network))
                node.ReallocateMemoryBlocks();
        }

        public override bool UpdateMemoryModel(MyProject project, List<MyNode> orderedNodes)
        {
            if (!orderedNodes.Any())
            {
                return true;
            }

            int attempts = 0;
            bool anyOutputChanged = false;

            try
            {
                while (attempts < MAX_BLOCKS_UPDATE_ATTEMPTS)
                {
                    attempts++;
                    anyOutputChanged = false;

                    anyOutputChanged |= UpdateAndCheckChange(project.World);
                    orderedNodes.ForEach(node => anyOutputChanged |= UpdateAndCheckChange(node));

                    if (!anyOutputChanged)
                    {
                        //MyLog.INFO.WriteLine("Successful update after " + attempts + " cycle(s).");
                        break;
                    }
                }
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Exception occured while updating memory model: " + e.Message);
                throw;
            }

            return anyOutputChanged;
        }

        private bool UpdateAndCheckChange(MyNode node)
        {
            node.PushOutputBlockSizes();
            node.UpdateMemoryBlocks();
            return node.AnyOutputSizeChanged();
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

        private void LoadBlocks(IEnumerable<MyWorkingNode> nodeList)
        {
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();

            var isFirst = true;
            foreach (MyWorkingNode node in nodeList)
            {
                if (LoadAllNodesData || node.LoadOnStart)
                {
                    foreach (MyAbstractMemoryBlock mb in MyMemoryManager.Instance.GetBlocks(node))
                    {
                        if (mb.Persistable)
                        {
                            if (isFirst)
                                MyLog.INFO.WriteLine("Loading state from: " + MyMemoryBlockSerializer.GetTempStorage(m_project));

                            isFirst = false;
                            serializer.LoadBlock(mb, GlobalDataFolder);
                        }
                    }
                }
            }
        }

        private void SaveBlocks(IEnumerable<MyWorkingNode> nodeList)
        {
            MyMemoryBlockSerializer serializer = new MyMemoryBlockSerializer();

            var isFirst = true;
            foreach (MyWorkingNode node in nodeList)
            {
                if (SaveAllNodesData || node.SaveOnStop)
                {
                    foreach (MyAbstractMemoryBlock mb in MyMemoryManager.Instance.GetBlocks(node))
                    {
                        if (mb.Persistable)
                        {
                            if (isFirst)
                                MyLog.INFO.WriteLine("Saving state to: " + MyMemoryBlockSerializer.GetTempStorage(m_project));

                            isFirst = false;
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
        public MySimulationException(string message) : base(message) { }

        public MySimulationException(string message, Exception innerException) : base(message, innerException) { }
    }
 }
