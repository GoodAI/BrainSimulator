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

        public TWInitTask InitTask { get; private set; }
        public TWGetInputTask GetInputTask { get; private set; }
        public TWUpdateTask UpdateTask { get; private set; }

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

        private IFovAvatarRR FovRR { get; set; }
        private IFofAvatarRR FofRR { get; set; }
        private IFreeMapRR FreeRR { get; set; }


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
            validator.AssertError(FoFMultisampleLevel >= 0 && FoFMultisampleLevel <= 4, this, "Multisample level must be between zero and five.");
            validator.AssertError(FoVMultisampleLevel >= 0 && FoVMultisampleLevel <= 4, this, "Multisample level must be between zero and five.");
            validator.AssertError(FreeViewMultisampleLevel >= 0 && FreeViewMultisampleLevel <= 4, this, "Multisample level must be between zero and five.");
            validator.AssertWarning(
                FoFMultisampleLevel == 1 || FoVMultisampleLevel == 1 || FreeViewMultisampleLevel == 1,
                this, "Multisample level of 1 is the same as level 2 (4x MSAA). Don't ask why.");

            if (Controls != null)
                validator.AssertError(Controls.Count >= 84 || Controls.Count == m_controlsCount, this, "Controls size has to be of size " + m_controlsCount + " or 84+. Use device input node for controls, or provide correct number of inputs");
        }

        private static string GetDllDirectory()
        {
            return Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        }

        public override void UpdateMemoryBlocks()
        {
            if (!File.Exists(SaveFile) || !File.Exists(TilesetTable) || FoFSize <= 0 || FoVSize <= 0 || Width <= 0 || Height <= 0 || ResolutionWidth <= 0 || ResolutionHeight <= 0 || FoFResHeight <= 0 || FoFResWidth <= 0 || FoVResHeight <= 0 || FoVResWidth <= 0)
                return;

            foreach (MyMemoryBlock<float> memBlock in new[] { VisualFov, VisualFof, VisualFree })
            {
                memBlock.Unmanaged = !CopyDataThroughCPU;
                memBlock.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.Raw;
            }

            VisualFov.Dims = new TensorDimensions(FoVResWidth, FoVResHeight);
            VisualFof.Dims = new TensorDimensions(FoFResWidth, FoFResHeight);
            VisualFree.Dims = new TensorDimensions(ResolutionWidth, ResolutionHeight);

            Text.Count = MaxMessageLength;
        }
    }
}
