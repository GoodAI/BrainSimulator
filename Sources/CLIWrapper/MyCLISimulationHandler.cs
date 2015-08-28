using GoodAI.Core;
using GoodAI.Core.Execution;
using GoodAI.Core.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLIWrapper
{
    public class MyCLISimulationHandler
    {
        public enum SimulationState
        {
            [Description("Running")]
            RUNNING,
            [Description("Paused")]
            PAUSED,
            [Description("Design Mode")]
            STOPPED,
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
                    m_simulation.Clear();
                    m_simulation.Finish();
                }
                m_simulation = value;
            }
        }

        private SimulationState m_state;
        public SimulationState State
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
        public static List<Tuple<int, uint, MonitorFunc>> SimulationMonitors { get; private set; }
        public static Hashtable MonitorResults;
        public delegate float[] MonitorFunc(MySimulation simulation);

        public MyCLISimulationHandler()
        {
            State = SimulationState.STOPPED;
            SimulationMonitors = new List<Tuple<int, uint, MonitorFunc>>();
            MonitorResults = new Hashtable();
        }

        public void AddMonitor(Tuple<int, uint, MonitorFunc> x)
        {
            SimulationMonitors.Add(x);
            MonitorResults[x.Item1] = new List<float[]>();
        }

        public void StartSimulation(uint steps, uint logStep)
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

            State = SimulationState.RUNNING;
            MyKernelFactory.Instance.SetCurrent(MyKernelFactory.Instance.DevCount - 1);

            int speedStart = Environment.TickCount;
            uint speedStep = SimulationStep;
            for (int i = 0; i < steps; ++i)
            {
                Simulation.ExecuteStep();

                foreach (Tuple<int, uint, MonitorFunc> m in SimulationMonitors)
                {
                    if (SimulationStep % m.Item2 == 0)
                    {
                        float[] value = (float[])m.Item3(Simulation);
                        (MonitorResults[m.Item1] as List<float[]>).Add(value);
                    }
                }

                if (logStep > 0 && i % logStep == 0)
                {
                    float SimulationSpeed = (SimulationStep - speedStep) * 1000.0f / (Environment.TickCount - speedStart);
                    MyLog.INFO.WriteLine("[" + SimulationStep + "] Running at " + SimulationSpeed + "/s");
                    speedStart = Environment.TickCount;
                    speedStep = SimulationStep;
                }
            }

            State = SimulationState.PAUSED;
        }

        public Hashtable Results()
        {
            return MonitorResults;
        }

        public void StopSimulation()
        {
            MyLog.INFO.WriteLine("Cleaning up world...");
            Project.World.Cleanup();

            MyLog.INFO.WriteLine("Freeing memory...");
            Simulation.FreeMemory();
            PrintMemoryInfo();

            MyLog.INFO.WriteLine("Clearing simulation...");
            Simulation.Clear();

            SimulationMonitors.Clear();
            MonitorResults.Clear();
            MyLog.INFO.WriteLine("Stopped.");
            State = SimulationState.STOPPED;
        }

        public void Finish()
        {
            Simulation = null;
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
        public event StateChangedHandler StateChanged;
        public event ProgressChangedEventHandler ProgressChanged;
    }
}
