using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Control;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Design;
using System.IO;
using System.Windows.Forms.Design;
using ToyWorldFactory;
using YAXLib;

namespace GoodAI.ToyWorld
{
    public partial class ToyWorld : MyWorld
    {
        private readonly int m_controlsCount = 13;

        public TWUpdateTask UpdateTask { get; private set; }

        public TWGetInputTask GetInputTask { get; private set; }

        public event EventHandler WorldInitialized = delegate { };


        #region Memblocks

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

        #endregion

        #region BrainSim properties

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

        [MyBrowsable, Category("FoF view"), DisplayName("Multisample level")]
        [YAXSerializableField(DefaultValue = 2)]
        public int FoFMultisampleLevel { get; set; }


        [MyBrowsable, Category("FoV view"), DisplayName("FoV size")]
        [YAXSerializableField(DefaultValue = 21)]
        public int FoVSize { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("FoV resolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoVResWidth { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("FoV resolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoVResHeight { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("Multisample level")]
        [YAXSerializableField(DefaultValue = 2)]
        public int FoVMultisampleLevel { get; set; }


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

        [MyBrowsable, Category("Free view"), DisplayName("\tResolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int ResolutionWidth { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tResolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int ResolutionHeight { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("Multisample level")]
        [YAXSerializableField(DefaultValue = 2)]
        public int FreeViewMultisampleLevel { get; set; }


        [MyBrowsable, DisplayName("Maximum message length")]
        [YAXSerializableField(DefaultValue = 128)]
        public int MaxMessageLength { get; set; }

        #endregion


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
            validator.AssertError(Controls != null, this, "No controls available");

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
            validator.AssertError(FoFMultisampleLevel >= 0 && FoFMultisampleLevel <= 5, this, "Multisample level must be between zero and five.");
            validator.AssertError(FoVMultisampleLevel >= 0 && FoVMultisampleLevel <= 5, this, "Multisample level must be between zero and five.");
            validator.AssertError(FreeViewMultisampleLevel >= 0 && FreeViewMultisampleLevel <= 5, this, "Multisample level must be between zero and five.");

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
                    rr.MultisampleLevel = FoVMultisampleLevel;
                });

            m_fofRR = ObtainRR<IFofAvatarRR>(VisualFof, myAvatarId,
                rr =>
                {
                    rr.FovAvatarRenderRequest = m_fovRR;
                    rr.Size = new SizeF(FoFSize, FoFSize);
                    rr.Resolution = new Size(FoFResWidth, FoFResHeight);
                    rr.MultisampleLevel = FoFMultisampleLevel;
                });

            m_freeRR = ObtainRR<IFreeMapRR>(VisualFree,
                rr =>
                {
                    rr.Size = new SizeF(Width, Height);
                    rr.Resolution = new Size(ResolutionWidth, ResolutionHeight);
                    rr.MultisampleLevel = FreeViewMultisampleLevel;
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

            rr.FlipYAxis = true;

            rr.CopyImageThroughCpu = CopyDataThroughCPU;
            targetMemBlock.ExternalPointer = 0; // first reset ExternalPointer

            if (!CopyDataThroughCPU)
            {
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
    }
}
