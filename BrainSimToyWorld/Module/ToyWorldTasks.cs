using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Control;
using Logger;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ToyWorldFactory;

namespace GoodAI.ToyWorld
{
    public partial class ToyWorld
    {
        [MyTaskInfo(OneShot = true, Disabled = false)]
        public class TWInitTask : MyTask<ToyWorld>
        {
            public override void Init(int nGPU)
            {
                if (Owner.GameCtrl != null)
                    Owner.GameCtrl.Dispose(); // Should dispose RRs and controllers too


                GameSetup setup = new GameSetup(
                    new FileStream(Owner.SaveFile, FileMode.Open, FileAccess.Read, FileShare.Read),
                    new StreamReader(Owner.TilesetTable));
                Owner.GameCtrl = GameFactory.GetThreadSafeGameController(setup);
                Owner.GameCtrl.Init();

                int[] avatarIds = Owner.GameCtrl.GetAvatarIds();
                if (avatarIds.Length == 0)
                {
                    MyLog.ERROR.WriteLine("No avatar found in map!");
                    return;
                }


                // Setup controllers
                int myAvatarId = avatarIds[0];
                Owner.AvatarCtrl = Owner.GameCtrl.GetAvatarController(myAvatarId);

                // Setup render requests
                Owner.FovRR = ObtainRR<IFovAvatarRR>(Owner.VisualFov, myAvatarId,
                    rr =>
                    {
                        rr.Size = new SizeF(Owner.FoVSize, Owner.FoVSize);
                        rr.Resolution = new Size(Owner.FoVResWidth, Owner.FoVResHeight);
                        rr.MultisampleLevel = Owner.FoVMultisampleLevel;
                        rr.DrawNoise = Owner.DrawNoise;
                        rr.NoiseIntensityCoefficient = Owner.NoiseIntensity;
                        rr.DrawSmoke = Owner.DrawSmoke;
                        rr.SmokeIntensityCoefficient = Owner.SmokeIntensity;
                        rr.SmokeScaleCoefficient = Owner.SmokeScale;
                        rr.SmokeTransformationSpeedCoefficient = Owner.SmokeTransformationSpeed;
                    });

                Owner.FofRR = ObtainRR<IFofAvatarRR>(Owner.VisualFof, myAvatarId,
                    rr =>
                    {
                        rr.FovAvatarRenderRequest = Owner.FovRR;
                        rr.Size = new SizeF(Owner.FoFSize, Owner.FoFSize);
                        rr.Resolution = new Size(Owner.FoFResWidth, Owner.FoFResHeight);
                        rr.MultisampleLevel = Owner.FoFMultisampleLevel;
                        rr.DrawNoise = Owner.DrawNoise;
                        rr.NoiseIntensityCoefficient = Owner.NoiseIntensity;
                        rr.DrawSmoke = Owner.DrawSmoke;
                        rr.SmokeIntensityCoefficient = Owner.SmokeIntensity;
                        rr.SmokeScaleCoefficient = Owner.SmokeScale;
                        rr.SmokeTransformationSpeedCoefficient = Owner.SmokeTransformationSpeed;
                    });

                Owner.FreeRR = ObtainRR<IFreeMapRR>(Owner.VisualFree,
                    rr =>
                    {
                        rr.Size = new SizeF(Owner.Width, Owner.Height);
                        rr.Resolution = new Size(Owner.ResolutionWidth, Owner.ResolutionHeight);
                        rr.MultisampleLevel = Owner.FreeViewMultisampleLevel;
                        rr.SetPositionCenter(Owner.CenterX, Owner.CenterY);
                        // no noise or smoke, this view is for the researcher
                    });

                Owner.WorldInitialized(this, EventArgs.Empty);
            }

            private T InitRR<T>(T rr, MyMemoryBlock<float> targetMemBlock, Action<T> initializer = null) where T : class, IRenderRequestBase
            {
                // Setup the render request properties
                rr.GatherImage = true;

                if (initializer != null)
                    initializer.Invoke(rr);

                rr.FlipYAxis = true;

                rr.CopyImageThroughCpu = Owner.CopyDataThroughCPU;
                targetMemBlock.ExternalPointer = 0; // first reset ExternalPointer

                if (Owner.CopyDataThroughCPU)
                    return rr;

                // Setup data copying to our unmanaged memblocks
                uint renderTextureHandle = 0;
                CudaOpenGLBufferInteropResource renderResource = null;

                rr.OnPreRenderingEvent += (sender, vbo) =>
                {
                    if (renderResource != null && renderResource.IsMapped)
                        renderResource.UnMap();
                };

                rr.OnPostRenderingEvent += (sender, vbo) =>
                {
                    // Vbo can be allocated during drawing, create the resource after that
                    MyKernelFactory.Instance.GetContextByGPU(Owner.GPU).SetCurrent();

                    if (renderResource == null || vbo != renderTextureHandle)
                    {
                        if (renderResource != null)
                            renderResource.Dispose();

                        renderTextureHandle = vbo;
                        renderResource = new CudaOpenGLBufferInteropResource(renderTextureHandle,
                            CUGraphicsRegisterFlags.ReadOnly); // Read only by CUDA
                    }

                    renderResource.Map();
                    targetMemBlock.ExternalPointer = renderResource.GetMappedPointer<uint>().DevicePointer.Pointer;
                    targetMemBlock.FreeDevice();
                    targetMemBlock.AllocateDevice();
                };


                // Initialize the target memory block
                targetMemBlock.ExternalPointer = 1;
                // Use a dummy number that will get replaced on first Execute call to suppress MemBlock error during init

                return rr;
            }

            private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, int avatarId, Action<T> initializer = null) where T : class, IAvatarRenderRequest
            {
                T rr = Owner.GameCtrl.RegisterRenderRequest<T>(avatarId);
                return InitRR(rr, targetMemBlock, initializer);
            }

            private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, Action<T> initializer = null) where T : class, IRenderRequest
            {
                T rr = Owner.GameCtrl.RegisterRenderRequest<T>();
                return InitRR(rr, targetMemBlock, initializer);
            }

            public override void Execute()
            { }
        }

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
                    TransferFromRRToMemBlock(Owner.FovRR, Owner.VisualFov);
                    TransferFromRRToMemBlock(Owner.FofRR, Owner.VisualFof);
                    TransferFromRRToMemBlock(Owner.FreeRR, Owner.VisualFree);
                }

                ObtainMessageFromBrain();
                SendMessageToBrain();

                ObtainSignals();
            }

            private void ObtainSignals()
            {
                int i = 0;
                foreach (KeyValuePair<string, float> signal in Owner.GameCtrl.GetSignals())
                {
                    int offset = Owner.OutputBranches - Owner.GameCtrl.GetSignals().Count;
                    Owner.Owner.Network.GroupInputNodes[offset + i].Name = signal.Key;
                    Owner.Owner.Network.GroupInputNodes[offset + i].Updated();

                    (Owner.m_outputs[offset + i] as MyMemoryBlock<float>).Host[0] = signal.Value;
                    Owner.m_outputs[offset + i].SafeCopyToDevice();
                    i++;
                }
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
