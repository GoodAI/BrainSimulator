using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Control;
using Logger;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GoodAI.ToyWorld
{
    public partial class ToyWorld
    {
        /// <summary>
        /// Initializes the ToyWorld core and sets up render requests.
        /// </summary>
        [MyTaskInfo(OneShot = true, Disabled = false)]
        public class TWInitTask : MyTask<ToyWorld>
        {
            public override void Init(int nGPU)
            {
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
                GameObjectSettings gameObjects = new GameObjectSettings(RenderRequestGameObject.TileLayers | RenderRequestGameObject.ObjectLayers)
                {
                    Use3D = Owner.Use3D
                };
                EffectSettings? effects = null;
                PostprocessingSettings? post = null;
                // Overlays are not used for now (no BrainSim property to switch them on) because there is a separate renderrequest for inventory Tool

                // Setup effects
                RenderRequestEffect enabledEffects = RenderRequestEffect.None;

                if (Owner.EnableDayAndNightCycle)
                    enabledEffects |= RenderRequestEffect.DayNight;
                if (Owner.DrawLights)
                    enabledEffects |= RenderRequestEffect.Lights;
                if (Owner.DrawSmoke)
                    enabledEffects |= RenderRequestEffect.Smoke;

                if (enabledEffects != RenderRequestEffect.None)
                    effects = new EffectSettings(enabledEffects)
                    {
                        SmokeIntensityCoefficient = Owner.SmokeIntensity,
                        SmokeScaleCoefficient = Owner.SmokeScale,
                        SmokeTransformationSpeedCoefficient = Owner.SmokeTransformationSpeed,
                    };

                // Setup postprocessing
                RenderRequestPostprocessing enabledPostprocessing = RenderRequestPostprocessing.None;

                if (Owner.DrawNoise)
                    enabledPostprocessing |= RenderRequestPostprocessing.Noise;

                if (enabledPostprocessing != RenderRequestPostprocessing.None)
                    post = new PostprocessingSettings(enabledPostprocessing)
                    {
                        NoiseIntensityCoefficient = Owner.NoiseIntensity,
                    };


                // Get and setup RRs
                Owner.FovRR = ObtainRR<IFovAvatarRR>(Owner.VisualFov, Owner.VisualFovDepth, myAvatarId,
                    rr =>
                    {
                        rr.Size = new SizeF(Owner.FoVSize, Owner.FoVSize);
                        rr.Resolution = new Size(Owner.FoVResWidth, Owner.FoVResHeight);
                        rr.MultisampleLevel = Owner.FoVMultisampleLevel;
                        rr.RotateMap = Owner.RotateMap;
                        rr.GameObjects = gameObjects;
                        rr.Effects = effects ?? new EffectSettings(RenderRequestEffect.None);
                        rr.Postprocessing = post ?? new PostprocessingSettings(RenderRequestPostprocessing.None);
                    });

                Owner.FofRR = ObtainRR<IFofAvatarRR>(Owner.VisualFof, Owner.VisualFofDepth, myAvatarId,
                    rr =>
                    {
                        rr.FovAvatarRenderRequest = Owner.FovRR;
                        rr.Size = new SizeF(Owner.FoFSize, Owner.FoFSize);
                        rr.Resolution = new Size(Owner.FoFResWidth, Owner.FoFResHeight);
                        rr.MultisampleLevel = Owner.FoFMultisampleLevel;
                        rr.RotateMap = Owner.RotateMap;
                        rr.GameObjects = gameObjects;
                        rr.Effects = effects ?? new EffectSettings(RenderRequestEffect.None);
                        rr.Postprocessing = post ?? new PostprocessingSettings(RenderRequestPostprocessing.None);
                    });

                Owner.FreeRR = ObtainRR<IFreeMapRR>(Owner.VisualFree, Owner.VisualFreeDepth,
                    rr =>
                    {
                        rr.Size = new SizeF(Owner.Width, Owner.Height);
                        rr.Resolution = new Size(Owner.ResolutionWidth, Owner.ResolutionHeight);
                        rr.MultisampleLevel = Owner.FreeViewMultisampleLevel;
                        rr.SetPositionCenter(Owner.CenterX, Owner.CenterY);
                        rr.GameObjects = gameObjects;
                        // no noise, smoke, postprocessing or overlays -- this view is for the researcher
                    });

                Owner.ToolRR = ObtainRR<IToolAvatarRR>(Owner.VisualTool, null, myAvatarId,
                    rr =>
                    {
                        rr.Size = new SizeF(Owner.ToolSize, Owner.ToolSize);
                        rr.Resolution = new Size(Owner.ToolResWidth, Owner.ToolResHeight);
                        rr.Overlay = new OverlaySettings(RenderRequestOverlay.InventoryTool)
                        {
                            ToolBackground = Owner.ToolBackgroundType,
                        };
                        // None of the other settings have any effect
                    });


                Owner.WorldInitialized(this, EventArgs.Empty);
            }

            private T InitRR<T>(T rr, MyMemoryBlock<float> targetMemBlock, MyMemoryBlock<float> targetDepthMemBlock, Action<T> initializer = null)
                where T : class, IRenderRequestBase
            {
                if (initializer != null)
                    initializer.Invoke(rr);

                rr.FlipYAxis = true;


                targetMemBlock.ExternalPointer = 0; // first reset ExternalPointer

                // Setup image copying from RR through Cpu
                if (Owner.CopyDataThroughCPU)
                {
                    ImageSettings imageSettings = new ImageSettings(RenderRequestImageCopyingMode.Cpu)
                    {
                        CopyDepth = Owner.CopyDepthData && targetDepthMemBlock != null,
                    };

                    imageSettings.OnSceneBufferPrepared += (request, data, depthData) =>
                    {
                        int width = rr.Resolution.Width;
                        int stride = width * sizeof(uint);
                        int lines = data.Length / width;

                        for (int i = 0; i < lines; ++i)
                            Buffer.BlockCopy(data, i * stride, targetMemBlock.Host, i * width * sizeof(uint), stride);

                        if (imageSettings.CopyDepth)
                            for (int i = 0; i < lines; ++i)
                                Buffer.BlockCopy(depthData, i * stride, targetDepthMemBlock.Host, i * width * sizeof(float), stride);

                        // targetMemBlock.SafeCopyToDevice(); this needs to be called on the BrainSim thread
                        // targetDepthMemBlock.SafeCopyToDevice(); this needs to be called on the BrainSim thread
                    };

                    rr.Image = imageSettings;
                    return rr;
                }


                // Setup image copying from RR through Pbo
                ImageSettings image = new ImageSettings(RenderRequestImageCopyingMode.OpenglPbo)
                {
                    CopyDepth = Owner.CopyDepthData && targetDepthMemBlock != null,
                };

                // Setup data copying to our unmanaged memblocks
                uint renderTextureHandle = 0;
                uint depthRenderTextureHandle = 0;
                CudaOpenGLBufferInteropResource renderResource = null;
                CudaOpenGLBufferInteropResource depthRenderResource = null;

                image.OnPreRenderingEvent += (sender, vbo, depthVbo) =>
                {
                    if (renderResource != null && renderResource.IsMapped)
                        renderResource.UnMap();

                    if (image.CopyDepth)
                        if (depthRenderResource != null && depthRenderResource.IsMapped)
                            depthRenderResource.UnMap();
                };

                image.OnPostRenderingEvent += (sender, vbo, depthVbo) =>
                {
                    // Vbo can be allocated during drawing, create the resource after that (post-rendering)
                    MyKernelFactory.Instance.GetContextByGPU(Owner.GPU).SetCurrent();

                    // Fill color memblock
                    if (renderResource == null || vbo != renderTextureHandle)
                    {
                        if (renderResource != null)
                            renderResource.Dispose();

                        renderTextureHandle = vbo;
                        try
                        {
                            renderResource = new CudaOpenGLBufferInteropResource(renderTextureHandle,
                                image.CopyDepth ? CUGraphicsRegisterFlags.None : CUGraphicsRegisterFlags.ReadOnly); // Read only by CUDA
                        }
                        catch (Exception e)
                        {
                            MyLog.ERROR.WriteLine("calling CudaOpenGLBufferInteropResource returns " + e +
                                ". Go to World properties and in Runtime section set Copy data through CPU to True");
                            throw e;
                        }
                    }

                    renderResource.Map();
                    targetMemBlock.ExternalPointer = renderResource.GetMappedPointer<uint>().DevicePointer.Pointer;
                    targetMemBlock.FreeDevice();
                    targetMemBlock.AllocateDevice();

                    // Fill depth memblock
                    if (image.CopyDepth)
                    {
                        if (depthRenderResource == null || depthVbo != depthRenderTextureHandle)
                        {
                            if (depthRenderResource != null)
                                depthRenderResource.Dispose();

                            depthRenderTextureHandle = depthVbo;
                            depthRenderResource = new CudaOpenGLBufferInteropResource(
                                depthRenderTextureHandle,
                                CUGraphicsRegisterFlags.ReadOnly); // Read only by CUDA
                        }

                        depthRenderResource.Map();
                        targetDepthMemBlock.ExternalPointer = depthRenderResource.GetMappedPointer<float>().DevicePointer.Pointer;
                        targetDepthMemBlock.FreeDevice();
                        targetDepthMemBlock.AllocateDevice();
                    }
                };

                rr.Image = image;


                // Initialize the target memory block
                // Use a dummy number that will get replaced on first Execute call to suppress MemBlock error during init
                targetMemBlock.ExternalPointer = 1;

                if (targetDepthMemBlock != null)
                    targetDepthMemBlock.ExternalPointer = 1;

                return rr;
            }

            private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, MyMemoryBlock<float> depthTargetMemBlock, int avatarId, Action<T> initializer = null)
                where T : class, IAvatarRenderRequest
            {
                T rr = Owner.GameCtrl.RegisterRenderRequest<T>(avatarId);
                return InitRR(rr, targetMemBlock, depthTargetMemBlock, initializer);
            }

            private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, MyMemoryBlock<float> depthTargetMemBlock, Action<T> initializer = null)
                where T : class, IRenderRequest
            {
                T rr = Owner.GameCtrl.RegisterRenderRequest<T>();
                return InitRR(rr, targetMemBlock, depthTargetMemBlock, initializer);
            }

            public override void Execute()
            { }
        }

        /// <summary>
        /// Encodes Brain Simulator control outputs and provides them to the ToyWorld's avatar.
        /// </summary>
        public class TWGetInputTask : MyTask<ToyWorld>
        {

            public override void Init(int nGPU) { }

            public override void Execute()
            {
                if (SimulationStep != 0 && SimulationStep % Owner.RunEvery != 0)
                    return;

                Owner.Controls.SafeCopyToHost();

                float leftSignal = Owner.Controls.Host[ControlMapper.Idx("left")];
                float rightSignal = Owner.Controls.Host[ControlMapper.Idx("right")];
                float fwSignal = Owner.Controls.Host[ControlMapper.Idx("forward")];
                float bwSignal = Owner.Controls.Host[ControlMapper.Idx("backward")];
                float rotLeftSignal = Owner.Controls.Host[ControlMapper.Idx("rot_left")];
                float rotRightSignal = Owner.Controls.Host[ControlMapper.Idx("rot_right")];

                float fof_left = Owner.Controls.Host[ControlMapper.Idx("fof_left")];
                float fof_right = Owner.Controls.Host[ControlMapper.Idx("fof_right")];
                float fof_up = Owner.Controls.Host[ControlMapper.Idx("fof_up")];
                float fof_down = Owner.Controls.Host[ControlMapper.Idx("fof_down")];

                float rotation = ConvertBiControlToUniControl(rotRightSignal, rotLeftSignal);
                float speed = ConvertBiControlToUniControl(fwSignal, bwSignal);
                float rightSpeed = ConvertBiControlToUniControl(rightSignal, leftSignal);
                float fof_x = ConvertBiControlToUniControl(fof_left, fof_right);
                float fof_y = ConvertBiControlToUniControl(fof_up, fof_down);

                bool interact = Owner.Controls.Host[ControlMapper.Idx("interact")] > 0.5;
                bool use = Owner.Controls.Host[ControlMapper.Idx("use")] > 0.5;
                bool pickup = Owner.Controls.Host[ControlMapper.Idx("pickup")] > 0.5;

                IAvatarControls ctrl = new AvatarControls(100, speed, rightSpeed, rotation, interact, use, pickup,
                    fof: new PointF(fof_x, fof_y));
                Owner.AvatarCtrl.SetActions(ctrl);
            }

            private static float ConvertBiControlToUniControl(float a, float b)
            {
                return a >= b ? a : -b;
            }
        }

        /// <summary>
        /// Performs a ToyWorld simulation step: updates game objects, resolves their actions, renders the scene and copies visual and other outputs to input memory blocks.
        /// </summary>
        public class TWUpdateTask : MyTask<ToyWorld>
        {
            private Stopwatch m_fpsStopwatch;
            private bool m_signalNodesNamed = false;

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
                ObtainActions();
                Owner.GameCtrl.FinishStep();

                if (Owner.CopyDataThroughCPU)
                {
                    Owner.VisualFov.SafeCopyToDevice();
                    Owner.VisualFof.SafeCopyToDevice();
                    Owner.VisualFree.SafeCopyToDevice();
                    Owner.VisualTool.SafeCopyToDevice();

                    if (Owner.CopyDepthData)
                    {
                        Owner.VisualFovDepth.SafeCopyToDevice();
                        Owner.VisualFofDepth.SafeCopyToDevice();
                        Owner.VisualFreeDepth.SafeCopyToDevice();
                    }
                }

                ObtainMessageFromBrain();
                SendMessageToBrain();
                ObtainSignals();
            }

            private void ObtainActions()
            {
                Dictionary<string, float> actions = Owner.AvatarCtrl.GetActions().ToDictionary();
                foreach (KeyValuePair<string, float> pair in actions)
                    Owner.ChosenActions.Host[ControlMapper.Idx(pair.Key)] = pair.Value;

                Owner.ChosenActions.SafeCopyToDevice();
            }

            private void ObtainSignals()
            {
                foreach (var item in Owner.GameCtrl.GetSignals().Select((signal, index) => new { signal, index }))
                {
                    // signals were note named yet AND Project's (Owner.Owner) World is equal to ToyWorld (Owner) - only in that case, there are GUI signal nodes
                    if (!m_signalNodesNamed && Owner.Owner.World == Owner)
                    {
                        Owner.GetSignalNode(item.index).Name = item.signal.Key;
                        Owner.GetSignalNode(item.index).Updated();
                    }

                    Owner.GetSignalMemoryBlock(item.index).Host[0] = item.signal.Value;
                    Owner.GetSignalMemoryBlock(item.index).SafeCopyToDevice();
                }

                m_signalNodesNamed = true;
            }

            private void SendMessageToBrain()
            {
                string message = Owner.AvatarCtrl.OutMessage;

                SetMessageTextBlock(message);
            }

            private void SetMessageTextBlock(string message)
            {
                for (int i = 0; i < Owner.Text.Count; ++i)
                    Owner.Text.Host[i] = 0;

                if (message == null)
                {
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
                Owner.AvatarCtrl.InMessage = string.Join("", Owner.TextIn.Host.Select(x => (char)x));
            }
        }
    }
}
