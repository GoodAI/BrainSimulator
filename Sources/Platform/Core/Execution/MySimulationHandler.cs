using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Threading;

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
                if (m_simulation != null && !m_simulation.IsFinished)
                    throw new InvalidOperationException("The simulation was not cleared. Call Finish() first.");
                m_simulation = value;
            }
        }

        private readonly BackgroundWorker m_worker;
        private bool doPause = false;

        private Action closeCallback = null;

        public int ReportInterval { get; set; } // How often should be speed of simulation reported
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

        private readonly int m_speedMeasureInterval;

        public float SimulationSpeed { get; private set; }

        private SimulationState m_state;
        private Action m_closeCallback;

        public SimulationState State    ///< State of the simulation
        { 
            get 
            { 
                return m_state; 
            }
            private set
            {
                SimulationState oldState = m_state;
                m_state = value;
                if (StateChanged != null)
                {
                    StateChanged(this, new StateEventArgs(oldState, m_state));
                }
                
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
            ReportInterval = 20;
            SleepInterval = 0;
            m_speedMeasureInterval = 2000;
            AutosaveInterval = 10000;
            Simulation = simulation;

            m_worker = new BackgroundWorker
            {
                WorkerReportsProgress = true,
                WorkerSupportsCancellation = true
            };

            m_worker.DoWork += m_worker_DoWork;
            m_worker.RunWorkerCompleted += m_worker_RunWorkerCompleted;
        }

        //UI thread
        /// <summary>
        /// Starts simulation
        /// </summary>
        /// <param name="oneStepOnly">Only one step of simulation is performed when true</param>
        public void StartSimulation(bool oneStepOnly)
        {
            if (State == SimulationState.STOPPED)
            {
                MyLog.INFO.WriteLine("Scheduling...");                
                Simulation.Schedule(Project);

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

            State = oneStepOnly ? SimulationState.RUNNING_STEP : SimulationState.RUNNING;

            MyKernelFactory.Instance.SetCurrent(MyKernelFactory.Instance.DevCount - 1);

            m_worker.RunWorkerAsync();
        }

        //UI thread
        public void StopSimulation()
        {
            if (State == SimulationState.RUNNING || State == SimulationState.RUNNING_STEP)
            {
                doPause = false;
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
            doPause = true;
            m_worker.CancelAsync();
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

            if (State == SimulationState.RUNNING_STEP)
            {
                try
                {
                    Simulation.PerformStep(true);
                }
                catch (Exception ex)
                {
                    MyLog.ERROR.WriteLine("Error occured during simulation: " + ex.Message);
                    e.Cancel = true;
                }

                if (ProgressChanged != null)
                {
                    ProgressChanged(this, null);
                }   
            }
            else if (State == SimulationState.RUNNING)
            {
                int start = Environment.TickCount;

                int speedStart = Environment.TickCount;
                uint speedStep = SimulationStep;

                while (!m_worker.CancellationPending)
                {
                    try
                    {
                        Simulation.PerformStep(false);
                    }
                    catch (Exception ex)
                    {
                        MyLog.ERROR.WriteLine("Error occured during simulation: " + ex.Message);
                        break;
                    }                 

                    if (SleepInterval > 0)
                    {
                        Thread.Sleep(SleepInterval);
                    }                                      

                    if (Environment.TickCount - speedStart > m_speedMeasureInterval)
                    {                                             
                        SimulationSpeed = (SimulationStep - speedStep) * 1000.0f / m_speedMeasureInterval;

                        speedStart = Environment.TickCount;
                        speedStep = SimulationStep;
                    }

                    if (Environment.TickCount - start > ReportInterval)
                    {
                        start = Environment.TickCount;

                        if (ProgressChanged != null)
                        {
                            ProgressChanged(this, null);
                        }
                    }
                }
                e.Cancel = true;
            }
            else
            {
                throw new IllegalStateException("Bad worker state: " + State);
            }
        }

        // NOT UI thread
        void m_worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Cancelled && !doPause)
            {
                DoStop();
            }
            else
            {
                MyLog.INFO.WriteLine("Paused.");
                State = SimulationState.PAUSED;
                Project.World.DoPause();
            }
        }

        private void DoStop()
        {
            if (m_closeCallback != null)
                Simulation.Finish();

            if (State != SimulationState.STOPPED)
            {
                MyLog.INFO.WriteLine("Cleaning up world...");
                Project.World.Cleanup();

                MyLog.INFO.WriteLine("Freeing memory...");
                Simulation.FreeMemory();
                PrintMemoryInfo();

                MyKernelFactory.Instance.RecoverContexts();

                MyLog.INFO.WriteLine("Clearing simulation...");
                Simulation.Clear();            

                MyLog.INFO.WriteLine("Stopped after "+this.SimulationStep+" steps.");
                State = SimulationState.STOPPED;
            }

            // Cleanup and invoke the callback action.
            if (m_closeCallback != null)
                m_closeCallback();
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
        /// Emmited each ReportInterval, or when only one step of simulation is ran
        /// </summary>
        public event ProgressChangedEventHandler ProgressChanged;
    }

    [Serializable]
    internal class IllegalStateException : Exception 
    {
        public IllegalStateException(string message) : base(message) { }
    }    
}
