using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Control;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Design;
using System.IO;
using System.Windows.Forms.Design;
using Logger;
using ToyWorldFactory;
using YAXLib;
using System.Diagnostics;
using System.Reflection;
using GoodAI.Core;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GoodAI.ToyWorld
{
    public class ToyWorld : MyWorld
    {
        private readonly int m_controlsCount = 13;

        public TWUpdateTask UpdateTask { get; private set; }

        public TWGetInputTask GetInputTask { get; private set; }

        public event EventHandler WorldInitialized = delegate { };

        [MyOutputBlock(0), MyUnmanaged]
        public MyMemoryBlock<float> VisualFov
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1), MyUnmanaged]
        public MyMemoryBlock<float> VisualFof
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2), MyUnmanaged]
        public MyMemoryBlock<float> VisualFree
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> Text
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Controls
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> TextIn
        {
            get { return GetInput(1); }
        }

        [MyBrowsable, Category("Runtime"), DisplayName("Run every Nth")]
        [YAXSerializableField(DefaultValue = 1)]
        public int RunEvery { get; set; }

        [MyBrowsable, Category("Runtime"), DisplayName("Use 60 FPS cap")]
        [YAXSerializableField(DefaultValue = false)]
        public bool UseFpsCap { get; set; }

        [MyBrowsable, Category("Runtime"), DisplayName("Copy data through CPU")]
        [YAXSerializableField(DefaultValue = false)]
        public bool CopyDataThroughCPU { get; set; }

        [MyBrowsable, Category("Files"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        [YAXSerializableField(DefaultValue = null), YAXCustomSerializer(typeof(MyPathSerializer))]
        public string TilesetTable { get; set; }

        [MyBrowsable, Category("Files"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        [YAXSerializableField(DefaultValue = null), YAXCustomSerializer(typeof(MyPathSerializer))]
        public string SaveFile { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("FoF size")]
        [YAXSerializableField(DefaultValue = 3)]
        public int FoFSize { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("FoF resolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoFResWidth { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("FoF resolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoFResHeight { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("FoV size")]
        [YAXSerializableField(DefaultValue = 21)]
        public int FoVSize { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("FoV resolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoVResWidth { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("FoV resolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoVResHeight { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tCenter - X")]
        [YAXSerializableField(DefaultValue = 25)]
        public float CenterX { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tCenter - Y")]
        [YAXSerializableField(DefaultValue = 25)]
        public float CenterY { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tWidth")]
        [YAXSerializableField(DefaultValue = 50)]
        public float Width { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tHeight")]
        [YAXSerializableField(DefaultValue = 50)]
        public float Height { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("Resolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int ResolutionWidth { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("Resolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int ResolutionHeight { get; set; }

        [MyBrowsable, DisplayName("Maximum message length")]
        [YAXSerializableField(DefaultValue = 128)]
        public int MaxMessageLength { get; set; }

        public IGameController GameCtrl { get; set; }
        public IAvatarController AvatarCtrl { get; set; }

        private IFovAvatarRR m_fovRR { get; set; }
        private IFofAvatarRR m_fofRR { get; set; }
        private IFreeMapRR m_freeRR { get; set; }

        public ToyWorld()
        {
            if (TilesetTable == null)
                TilesetTable = GetDllDirectory() + @"\res\GameActors\Tiles\Tilesets\TilesetTable.csv";
            if (SaveFile == null)
                SaveFile = GetDllDirectory() + @"\res\Worlds\mockup999_pantry_world.tmx";
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            validator.AssertError(File.Exists(SaveFile), this, "Please specify a correct SaveFile path in world properties.");
            validator.AssertError(File.Exists(TilesetTable), this, "Please specify a correct TilesetTable path in world properties.");

            validator.AssertError(FoFSize > 0, this, "FoF size has to be positive.");
            validator.AssertError(FoFResWidth > 0, this, "FoF resolution width has to be positive.");
            validator.AssertError(FoFResHeight > 0, this, "FoF resolution height has to be positive.");
            validator.AssertError(FoVSize > 0, this, "FoV size has to be positive.");
            validator.AssertError(FoVResWidth > 0, this, "FoV resolution width has to be positive.");
            validator.AssertError(FoVResHeight > 0, this, "FoV resolution height has to be positive.");
            validator.AssertError(Width > 0, this, "Free view width has to be positive.");
            validator.AssertError(Height > 0, this, "Free view height has to be positive.");
            validator.AssertError(ResolutionWidth > 0, this, "Free view resolution width has to be positive.");
            validator.AssertError(ResolutionHeight > 0, this, "Free view resolution height has to be positive.");

            if (Controls != null)
                validator.AssertError(Controls.Count >= 84 || Controls.Count == m_controlsCount, this, "Controls size has to be of size " + m_controlsCount + " or 84+. Use device input node for controls, or provide correct number of inputs");
        }

        public override void UpdateMemoryBlocks()
        {
            if (!File.Exists(SaveFile) || !File.Exists(TilesetTable) || FoFSize <= 0 || FoVSize <= 0 || Width <= 0 || Height <= 0 || ResolutionWidth <= 0 || ResolutionHeight <= 0 || FoFResHeight <= 0 || FoFResWidth <= 0 || FoVResHeight <= 0 || FoVResWidth <= 0)
                return;

            GameSetup setup = new GameSetup(new FileStream(SaveFile, FileMode.Open, FileAccess.Read, FileShare.Read), new StreamReader(TilesetTable));
            GameCtrl = GameFactory.GetThreadSafeGameController(setup);
            GameCtrl.Init();

            int[] avatarIds = GameCtrl.GetAvatarIds();
            if (avatarIds.Length == 0)
            {
                MyLog.ERROR.WriteLine("No avatar found in map!");
                return;
            }

            foreach (MyMemoryBlock<float> memBlock in new MyMemoryBlock<float>[3] { VisualFov, VisualFof, VisualFree })
            {
                memBlock.Unmanaged = !CopyDataThroughCPU;
                memBlock.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.Raw;
            }

            // Setup controllers
            int myAvatarId = avatarIds[0];
            AvatarCtrl = GameCtrl.GetAvatarController(myAvatarId);

            // Setup render requests
            m_fovRR = ObtainRR<IFovAvatarRR>(VisualFov, myAvatarId,
                rr =>
                {
                    rr.Size = new SizeF(FoVSize, FoVSize);
                    rr.Resolution = new Size(FoVResWidth, FoVResHeight);
                });

            m_fofRR = ObtainRR<IFofAvatarRR>(VisualFof, myAvatarId,
                rr =>
                {
                    rr.FovAvatarRenderRequest = m_fovRR;
                    rr.Size = new SizeF(FoFSize, FoFSize);
                    rr.Resolution = new Size(FoFResWidth, FoFResHeight);
                });

            m_freeRR = ObtainRR<IFreeMapRR>(VisualFree,
                rr =>
                {
                    rr.Size = new SizeF(Width, Height);
                    rr.Resolution = new Size(ResolutionWidth, ResolutionHeight);
                });
            m_freeRR.SetPositionCenter(CenterX, CenterY);

            WorldInitialized(this, EventArgs.Empty);

            Text.Count = MaxMessageLength;
        }

        private static string GetDllDirectory()
        {
            return Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        }

        private T InitRR<T>(T rr, MyMemoryBlock<float> targetMemBlock, Action<T> initializer = null) where T : class, IRenderRequestBase
        {
            // Setup the render request properties
            rr.GatherImage = true;

            if (initializer != null)
                initializer.Invoke(rr);

            rr.CopyImageThroughCpu = CopyDataThroughCPU;

            foreach (string memBlockName in new string[3] { "VisualFov", "VisualFof", "VisualFree" })
            {
                PropertyDescriptor desc = TypeDescriptor.GetProperties(this.GetType())[memBlockName];
                MyUnmanagedAttribute attr = (MyUnmanagedAttribute)desc.Attributes[typeof(MyUnmanagedAttribute)];
                PropertyInfo unmanaged = attr.GetType().GetProperty("Unmanaged");
                unmanaged.SetValue(attr, !CopyDataThroughCPU);
            }

            if (!CopyDataThroughCPU)
            {
                rr.FlipYAxis = true;

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
                    MyKernelFactory.Instance.GetContextByGPU(GPU).SetCurrent();

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
            }
            targetMemBlock.Dims = new TensorDimensions(rr.Resolution.Width, rr.Resolution.Height);
            return rr;
        }

        private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, int avatarId, Action<T> initializer = null) where T : class, IAvatarRenderRequest
        {
            T rr = GameCtrl.RegisterRenderRequest<T>(avatarId);
            return InitRR(rr, targetMemBlock, initializer);
        }

        private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, Action<T> initializer = null) where T : class, IRenderRequest
        {
            T rr = GameCtrl.RegisterRenderRequest<T>();
            return InitRR(rr, targetMemBlock, initializer);
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

                    m_controlIndexes["forward"] = 87;     // W
                    m_controlIndexes["backward"] = 83;    // S
                    m_controlIndexes["rot_left"] = 65;        // A
                    m_controlIndexes["rot_right"] = 68;       // D
                    m_controlIndexes["left"] = 81;        // Q
                    m_controlIndexes["right"] = 69;       // E

                    m_controlIndexes["fof_up"] = 73;      // I
                    m_controlIndexes["fof_left"] = 76;    // J
                    m_controlIndexes["fof_down"] = 75;    // K
                    m_controlIndexes["fof_right"] = 74;   // L

                    m_controlIndexes["interact"] = 66;    // B
                    m_controlIndexes["use"] = 78;         // N
                    m_controlIndexes["pickup"] = 77;      // M
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

                IAvatarControls ctrl = new AvatarControls(100, speed, rightSpeed, rotation, interact, use, pickup, fof: new PointF(fof_x, fof_y));
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
                    while (m_fpsStopwatch.Elapsed.Ticks < 166666) // a tick is 100 nanoseconds, 10000 ticks is 1 millisecond
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

                string message = Owner.AvatarCtrl.MessageIn;
                if (message == null)
                {
                    Owner.Text.Fill(0);
                    return;
                }
                for (int i = 0; i < message.Length; ++i)
                    Owner.Text.Host[i] = message[i];

                Owner.Text.SafeCopyToDevice();
                Owner.AvatarCtrl.MessageOut = null;

                Owner.TextIn.SafeCopyToHost();
            }

            private static void TransferFromRRToMemBlock(IRenderRequestBase rr, MyMemoryBlock<float> mb)
            {
                uint[] data = rr.Image;
                int width = rr.Resolution.Width;
                int stride = width * sizeof(uint);
                int lines = data.Length / width;

                for (int i = 0; i < lines; ++i)
                    Buffer.BlockCopy(data, i * stride, mb.Host, (mb.Count - (i + 1) * width) * sizeof(uint), stride);

                mb.SafeCopyToDevice();
            }
        }
    }
}
