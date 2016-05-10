using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Control;
using Logger;

namespace GoodAI.ToyWorld
{
    public partial class ToyWorld
    {
        public class TWGetInputTask : MyTask<ToyWorld>
        {
            private readonly Dictionary<string, int> m_controlIndexes = new Dictionary<string, int>();

            public override void Init(int nGPU)
            {
                if (Owner.Controls.Count == Owner.m_controlsCount)
                {
                    MyLog.INFO.WriteLine("ToyWorld: Controls set to WSAD mode.");

                    m_controlIndexes["forward"] = 0;
                    m_controlIndexes["backward"] = 1;
                    m_controlIndexes["left"] = 2;
                    m_controlIndexes["right"] = 3;
                    m_controlIndexes["rot_left"] = 4;
                    m_controlIndexes["rot_right"] = 5;
                    m_controlIndexes["fof_right"] = 6;
                    m_controlIndexes["fof_left"] = 7;
                    m_controlIndexes["fof_up"] = 8;
                    m_controlIndexes["fof_down"] = 9;
                    m_controlIndexes["interact"] = 10;
                    m_controlIndexes["use"] = 11;
                    m_controlIndexes["pickup"] = 12;
                }
                else if (Owner.Controls.Count >= 84)
                {
                    MyLog.INFO.WriteLine("ToyWorld: Controls set to keyboard mode.");

                    m_controlIndexes["forward"] = 87; // W
                    m_controlIndexes["backward"] = 83; // S
                    m_controlIndexes["rot_left"] = 65; // A
                    m_controlIndexes["rot_right"] = 68; // D
                    m_controlIndexes["left"] = 81; // Q
                    m_controlIndexes["right"] = 69; // E

                    m_controlIndexes["fof_up"] = 73; // I
                    m_controlIndexes["fof_left"] = 76; // J
                    m_controlIndexes["fof_down"] = 75; // K
                    m_controlIndexes["fof_right"] = 74; // L

                    m_controlIndexes["interact"] = 66; // B
                    m_controlIndexes["use"] = 78; // N
                    m_controlIndexes["pickup"] = 77; // M
                }
            }

            public override void Execute()
            {
                if (SimulationStep != 0 && SimulationStep % Owner.RunEvery != 0)
                    return;

                Owner.Controls.SafeCopyToHost();

                float leftSignal = Owner.Controls.Host[m_controlIndexes["left"]];
                float rightSignal = Owner.Controls.Host[m_controlIndexes["right"]];
                float fwSignal = Owner.Controls.Host[m_controlIndexes["forward"]];
                float bwSignal = Owner.Controls.Host[m_controlIndexes["backward"]];
                float rotLeftSignal = Owner.Controls.Host[m_controlIndexes["rot_left"]];
                float rotRightSignal = Owner.Controls.Host[m_controlIndexes["rot_right"]];

                float fof_left = Owner.Controls.Host[m_controlIndexes["fof_left"]];
                float fof_right = Owner.Controls.Host[m_controlIndexes["fof_right"]];
                float fof_up = Owner.Controls.Host[m_controlIndexes["fof_up"]];
                float fof_down = Owner.Controls.Host[m_controlIndexes["fof_down"]];

                float rotation = ConvertBiControlToUniControl(rotLeftSignal, rotRightSignal);
                float speed = ConvertBiControlToUniControl(fwSignal, bwSignal);
                float rightSpeed = ConvertBiControlToUniControl(leftSignal, rightSignal);
                float fof_x = ConvertBiControlToUniControl(fof_left, fof_right);
                float fof_y = ConvertBiControlToUniControl(fof_up, fof_down);

                bool interact = Owner.Controls.Host[m_controlIndexes["interact"]] > 0.5;
                bool use = Owner.Controls.Host[m_controlIndexes["use"]] > 0.5;
                bool pickup = Owner.Controls.Host[m_controlIndexes["pickup"]] > 0.5;

                IAvatarControls ctrl = new AvatarControls(100, speed, rightSpeed, rotation, interact, use, pickup,
                    fof: new PointF(fof_x, fof_y));
                Owner.AvatarCtrl.SetActions(ctrl);
            }

            private static float ConvertBiControlToUniControl(float a, float b)
            {
                return a >= b ? a : -b;
            }
        }

        public class TWUpdateTask : MyTask<ToyWorld>
        {
            private Stopwatch m_fpsStopwatch;

            public override void Init(int nGPU)
            {
                m_fpsStopwatch = Stopwatch.StartNew();
            }

            private static void PrintLogMessage(MyLog logger, TWLogMessage message)
            {
                logger.WriteLine("TWLog: " + message);
            }

            private static void PrintLogMessages()
            {
                foreach (TWLogMessage message in TWLog.GetAllLogMessages())
                {
                    switch (message.Severity)
                    {
                        case TWSeverity.Error:
                            {
                                PrintLogMessage(MyLog.ERROR, message);
                                break;
                            }
                        case TWSeverity.Warn:
                            {
                                PrintLogMessage(MyLog.WARNING, message);
                                break;
                            }
                        case TWSeverity.Info:
                            {
                                PrintLogMessage(MyLog.INFO, message);
                                break;
                            }
                        default:
                            {
                                PrintLogMessage(MyLog.DEBUG, message);
                                break;
                            }
                    }
                }
            }

            public override void Execute()
            {
                if (SimulationStep != 0 && SimulationStep % Owner.RunEvery != 0)
                    return;

                PrintLogMessages();

                if (Owner.UseFpsCap)
                {
                    // do a step at most every 16.6 ms, which leads to a 60FPS cap
                    while (m_fpsStopwatch.Elapsed.Ticks < 166666)
                    // a tick is 100 nanoseconds, 10000 ticks is 1 millisecond
                    {
                        ; // busy waiting for the next frame
                        // cannot use Sleep because it is too coarse (16ms)
                        // we need millisecond precision
                    }

                    m_fpsStopwatch.Restart();
                }

                Owner.GameCtrl.MakeStep();

                if (Owner.CopyDataThroughCPU)
                {
                    TransferFromRRToMemBlock(Owner.m_fovRR, Owner.VisualFov);
                    TransferFromRRToMemBlock(Owner.m_fofRR, Owner.VisualFof);
                    TransferFromRRToMemBlock(Owner.m_freeRR, Owner.VisualFree);
                }

                ObtainMessageFromBrain();
                SendMessageToBrain();
            }

            private void SendMessageToBrain()
            {
                string message = Owner.AvatarCtrl.MessageIn;
                if (message == null)
                {
                    Owner.Text.Fill(0);
                    return;
                }
                for (int i = 0; i < message.Length; ++i)
                    Owner.Text.Host[i] = message[i];

                Owner.Text.SafeCopyToDevice();
            }

            private void ObtainMessageFromBrain()
            {
                if (Owner.TextIn == null)
                    return;
                Owner.TextIn.SafeCopyToHost();
                Owner.AvatarCtrl.MessageOut = string.Join("", Owner.TextIn.Host.Select(x => (char)x));
            }

            private static void TransferFromRRToMemBlock(IRenderRequestBase rr, MyMemoryBlock<float> mb)
            {
                uint[] data = rr.Image;
                int width = rr.Resolution.Width;
                int stride = width * sizeof(uint);
                int lines = data.Length / width;

                for (int i = 0; i < lines; ++i)
                    Buffer.BlockCopy(data, i * stride, mb.Host, i * width * sizeof(uint), stride);

                mb.SafeCopyToDevice();
            }
        }
    }
}
