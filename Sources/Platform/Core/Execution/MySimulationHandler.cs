using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using GoodAI.Core.Nodes;
using System.Diagnostics;
using GoodAI.Platform.Core.Profiling;

namespace GoodAI.Core.Execution
{
    /// Managers MySimulation run
    public class MySimulationHandler : IDisposable
    {
        public enum SimulationState
        {
            [Description("Running")]
            RUNNING,
            [Description("Running")]
            RUNNING_STEP,
            [Description("Paused")]
            PAUSED,
            [Description("Design Mode")]
            STOPPED
        }
        
        public MyProject Project { get; set; }

        private MySimulation m_simulation;
        public MySimulation Simulation 
        {
            get { return m_simulation; }
            set
            {
                if (m_simulation != null)
                {
                    if (!m_simulation.IsFinished)
                        throw new InvalidOperationException("The simulation was not cleared. Call Finish() first.");

                    StateChanged -= m_simulation.OnStateChanged;
                }
                m_simulation = value;

                m_simulation.DebugTargetReached += OnDebugTargetReached;
                StateChanged += m_simulation.OnStateChanged;
            }
        }

        private readonly BackgroundWorker m_worker;
        private readonly ManualResetEvent m_workedCompleted;
        private bool m_doPause = false;

        public uint ReportIntervalSteps { get; set; } // How often (in steps) should be speed of simulation reported in RUNNING_STEP state
        public int ReportInterval { get; set; } // How often (in ms) should be speed of simulation reported in RUNNING state
        public int SleepInterval { get; set; }  // Amount of sleep (in ms) between two steps

        private bool m_autosaveEnabled;
        public bool AutosaveEnabled 
        {
            get { return m_autosaveEnabled; }
            set 
            { 
                m_autosaveEnabled = value;
                UpdateAutosaveInterval();
            }
        }

        private int m_autosaveInterval;
        public int AutosaveInterval 
        {
            get { return m_autosaveInterval; }
            set
            {
                m_autosaveInterval = value;
                UpdateAutosaveInterval();
            }
        }

        private void UpdateAutosaveInterval()
        {
            if (Simulation != null)
            {
                Simulation.AutoSaveInterval = AutosaveEnabled ? AutosaveInterval : 0;
            }
        }

        internal Exception m_simulationStoppedException;
        public class SimulationStoppedEventArgs : EventArgs
        {
            public Exception Exception { get; set; }
            public uint StepCount { get; set; }
        }

        public delegate void SimulationStoppedEventHandler(object sender, SimulationStoppedEventArgs args);
        public event SimulationStoppedEventHandler SimulationStopped;

        private readonly int m_speedMeasureIntervalMillisec;

        public float SimulationSpeed { get; private set; }

        public string SimulationSpeedStr
        {
            get
            {
                var s = SimulationSpeed;
                if (Math.Abs(s) < 0.000001)
                    return "?/s";

                return ((s < 10.0f)
                    ? $"{Math.Round(s, 1):0.0}"
                    : $"{Math.Round(s)}")
                    + "/s";
            }
        }

        private SimulationState m_state;
        private uint m_stepsToPerform;
        private Action m_closeCallback;
        private uint m_lastProgressChangedStep;

        public SimulationState State
        { 
            get 
            { 
                return m_state; 
            }
            private set
            {
                var oldState = m_state;
                m_state = value;

                StateChanged?.Invoke(this, new StateEventArgs(oldState, m_state));

            }
        }
        public uint SimulationStep { get { return Simulation.SimulationStep; } }    

        public bool CanStart { get { return State == SimulationState.STOPPED || State == SimulationState.PAUSED; } }    
        public bool CanStepOver { get { return CanStart; } }    
        public bool CanPause { get { return State == SimulationState.RUNNING; } }   
        public bool CanStop { get { return State == SimulationState.RUNNING || State == SimulationState.PAUSED; } }

        public bool CanStartDebugging { get { return State == SimulationState.STOPPED; } }
        public bool CanStepInto { get { return State == SimulationState.PAUSED && Simulation.InDebugMode; } }
        public bool CanStepOut { get { return State == SimulationState.PAUSED && Simulation.InDebugMode; } }

        //UI thread
        /// <summary>
        /// Constructor
        /// </summary>
        public MySimulationHandler(MySimulation simulation)
        {            
            State = SimulationState.STOPPED;
            SimulationSpeed = 0;
            ReportIntervalSteps = 100;
            ReportInterval = 20;
            SleepInterval = 0;
            m_speedMeasureIntervalMillisec = 998;  // Almost 1 second (to prevent some races).
            AutosaveInterval = 10000;

            Simulation = simulation;

            m_workedCompleted = new ManualResetEvent(true);

            m_worker = new BackgroundWorker
            {
                WorkerReportsProgress = true,
                WorkerSupportsCancellation = true
            };

            m_worker.DoWork += m_worker_DoWork;
            m_worker.RunWorkerCompleted += m_worker_RunWorkerCompleted;
        }

        /// <summary>
        /// Starts simulation.
        /// </summary>
        public void StartSimulation()
        {
            StartSimulation(stepCount: 0);
        }

        //UI thread
        /// <summary>
        /// Starts simulation for specified number of steps.
        /// </summary>
        /// <param name="stepCount">How many steps of simulation shall be performed (0 means unlimited).</param>
        public void StartSimulation(uint stepCount)
        {
            bool doFixedNumberOfSteps = (stepCount > 0);

            if (State == SimulationState.STOPPED)
            {
                MyLog.INFO.WriteLine("Scheduling...");                
                Simulation.Schedule(Project, new MyWorkingNode[] {Project.World, Project.Network});

                MyLog.INFO.WriteLine("Initializing tasks...");
                Simulation.Init();          

                MyLog.INFO.WriteLine("Allocating memory...");
                Simulation.AllocateMemory();
                PrintMemoryInfo();             

                MyLog.INFO.WriteLine("Starting simulation...");
            }
            else
            {
                MyLog.INFO.WriteLine("Resuming simulation...");
            }

            State = doFixedNumberOfSteps ? SimulationState.RUNNING_STEP : SimulationState.RUNNING;
            m_stepsToPerform = stepCount;
            m_lastProgressChangedStep = 0;

            // Clean up breakpoints.
            Simulation.CleanTemporaryBlockData();

            MyKernelFactory.Instance.SetCurrent(MyKernelFactory.Instance.DevCount - 1);

            m_workedCompleted.Reset();
            m_worker.RunWorkerAsync();
        }

        //UI thread
        public void StopSimulation()
        {
            if (State == SimulationState.RUNNING || State == SimulationState.RUNNING_STEP)
            {
                m_doPause = false;
                m_worker.CancelAsync();
            }
            else
            {
                DoStop();
            }
        }

        //UI thread
        public void PauseSimulation()
        {            
            m_doPause = true;
            m_worker.CancelAsync();
        }

        private void OnDebugTargetReached(object sender, EventArgs args)
        {
            PauseSimulation();
        }

        /// <summary>
        /// The closeCallback action is invoked after all of the cleanup is done.
        /// This is because the background thread cleanup cannot be done synchronously.
        /// </summary>
        /// <param name="closeCallback"></param>
        //UI thread
        public void Finish(Action closeCallback = null)
        {
            // If no callback is specified, use an empty one so that the sim still finishes.
            if (closeCallback == null)
                closeCallback = () => { };

            m_closeCallback = closeCallback;
            StopSimulation();
        }

        public void Dispose()
        {
            Finish();
        }

        //NOT in UI thread
        void m_worker_DoWork(object sender, DoWorkEventArgs e)
        {
            if (Thread.CurrentThread.Name == null)
                Thread.CurrentThread.Name = "Background Simulation Thread";

            if (State != SimulationState.RUNNING_STEP && State != SimulationState.RUNNING)
            {
                throw new IllegalStateException("Bad worker state: " + State);
            }

            Stopwatch stopwatch = Stopwatch.StartNew();
            long reportLastTicks = stopwatch.ElapsedTicks;
            long reportIntervalTicks = ReportInterval * Stopwatch.Frequency / 1000;

            uint performedSteps = 0;  // During debug, this counts IMyExecutable steps as opposed to whole sim steps.

            var runningAverage = new SimSpeedRunningAverage(minRecordIntervalMillisec: m_speedMeasureIntervalMillisec);

            SimulationSpeed = 0.0f;

            while (true)
            {
                if (State == SimulationState.RUNNING_STEP)
                {
                    if (performedSteps >= m_stepsToPerform)
                        break;
                }
                
                if (m_worker.CancellationPending)
                {
                    e.Cancel = true;
                    break;
                }

                try
                {
                    // If the simulation is in between two steps, we allow for model changes, block reallocation etc.
                    if (Simulation.IsStepFinished)
                    {
                        Simulation.PerformModelChanges();
                        Simulation.Reallocate();
                    }

                    Simulation.PerformStep(State == SimulationState.RUNNING_STEP);
                    ++performedSteps;

                    StepPerformed?.Invoke(this, null);
                }
                catch (Exception ex)
                {
                    MyLog.ERROR.WriteLine("Error occured during simulation: " + ex.Message);
                    m_simulationStoppedException = ex;
                    e.Cancel = true;
                    break;
                }

                if (SleepInterval > 0)
                {
                    Thread.Sleep(SleepInterval);
                }

                // Status reporting
                try
                {
                    bool doReportThisStep = (State == SimulationState.RUNNING_STEP)
                        && (performedSteps % ReportIntervalSteps == 0);

                    runningAverage.AddTimePoint(SimulationStep, doReportThisStep);

                    if (doReportThisStep || ((stopwatch.ElapsedTicks - reportLastTicks) >= reportIntervalTicks))
                    {
                        reportLastTicks = stopwatch.ElapsedTicks;

                        SimulationSpeed = runningAverage.GetItersPerSecond();

                        if (ProgressChanged != null)
                        {
                            m_lastProgressChangedStep = SimulationStep;
                            ProgressChanged(this, null);
                        }
                    }
                }
                catch (Exception ex)
                {
                    MyLog.WARNING.WriteLine("Error during sim. status reporting: " + ex.Message);
                    // Continue...
                }
            }
            
        }

        // NOT UI thread
        void m_worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (ProgressChanged != null && m_lastProgressChangedStep != SimulationStep)
            {
                ProgressChanged(this, null);
            }

            if (e.Cancelled && !m_doPause)
            {
                DoStop();
            }
            else
            {
                // This means we're either pausing, or this was a single simulation step.
                MyLog.INFO.WriteLine("Paused.");
                State = SimulationState.PAUSED;
            }

            m_workedCompleted.Set();
        }

        /// <summary>
        /// Blocks until the requested number of simulation steps had been performed.
        /// </summary>
        public void WaitUntilStepsPerformed()
        {
            m_workedCompleted.WaitOne();
        }

        // TODO: throw an exception if the model doesn't converge. The return value is unintuitive.
        /// <summary>
        /// Update the whole memory model - all blocks will get their memory block sizes updated correctly.
        /// Since this might not converge, only a set number of iterations is done.
        /// </summary>
        /// <returns>true if the model did not converge (error), false if it did.</returns>
        public bool UpdateMemoryModel()
        {            
            MyLog.INFO.WriteLine("Updating memory blocks...");

            List<MyNode> orderedNodes = OrderNetworkNodes(Project.Network);

            return Simulation.UpdateMemoryModel(Project, orderedNodes);
        }

        public static List<MyNode> OrderNetworkNodes(MyNodeGroup network)
        {
            IMyOrderingAlgorithm topoOps = new MyHierarchicalOrdering();
            return topoOps.EvaluateOrder(network);
        }

        public void RefreshTopologicalOrder()
        {
            OrderNetworkNodes(Project.Network);
        }



        private void DoStop()
        {
            // TODO(HonzaS): This is hacky, it needs to be redone properly.
            // 1) Stop the simulation if needed.
            // 2) Set the state to STOPPED => notifies the nodes to clean up.
            // 3) Clear everything else if we're quitting.
            var stopping = false;
            if (State != SimulationState.STOPPED)
            {
                stopping = true;
                MyLog.INFO.WriteLine("Cleaning up world...");
                Project.World.Cleanup();

                MyLog.INFO.WriteLine("Freeing memory...");
                Simulation.FreeMemory(didCrash: m_simulationStoppedException != null);
                PrintMemoryInfo();

                MyKernelFactory.Instance.RecoverContexts();

                // This needs to be set before Clear is called so that nodes can be notified about the state change.
                State = SimulationState.STOPPED;
            }

            if (m_closeCallback != null)
                Simulation.Finish();

            if (stopping)
            {
                MyLog.INFO.WriteLine("Clearing simulation...");
                // This will destroy the collection that holds the nodes, so it has to be the last thing.
                Simulation.Clear();
                MyLog.INFO.WriteLine("Stopped after "+this.SimulationStep+" steps.");

                if (SimulationStopped != null)
                {
                    var args = new SimulationStoppedEventArgs
                    {
                        Exception = m_simulationStoppedException,
                        StepCount = SimulationStep
                    };
                    SimulationStopped(this, args);
                }
            }

            // Cleanup and invoke the callback action.
            m_closeCallback?.Invoke();
        }

        private void PrintMemoryInfo() 
        {
            List<Tuple<SizeT, SizeT>> memInfos = MyKernelFactory.Instance.GetMemInfo();

            for (int i = 0; i < memInfos.Count; i++)
            {
                Tuple<SizeT, SizeT> memInfo = memInfos[i];
                SizeT used = memInfo.Item2 - memInfo.Item1;
                MyLog.INFO.WriteLine("GPU " + i + ": " + (used / 1024 / 1024) + " MB used,  " + (memInfo.Item1 / 1024 / 1024) + " MB free");
            }
        }

        public class StateEventArgs : EventArgs
        {
            public SimulationState OldState { get; internal set; }
            public SimulationState NewState { get; internal set; }

            public StateEventArgs(SimulationState oldState, SimulationState newState)
            {
                OldState = oldState;
                NewState = newState;
            }
        };
        public delegate void StateChangedHandler(object sender, StateEventArgs e);
        /// <summary>
        /// Emmited when simulation changes its SimulationState
        /// </summary>
        public event StateChangedHandler StateChanged;
        /// <summary>
        /// Emmited each ReportInterval/ReportIntervalSteps, or after the requested number of simulation steps had been performed
        /// </summary>
        public event ProgressChangedEventHandler ProgressChanged;
        /// <summary>
        /// Emmited after each simulation step
        /// </summary>
        public event ProgressChangedEventHandler StepPerformed;
    }

    [Serializable]
    internal class IllegalStateException : Exception 
    {
        public IllegalStateException(string message) : base(message) { }
    }    
}
