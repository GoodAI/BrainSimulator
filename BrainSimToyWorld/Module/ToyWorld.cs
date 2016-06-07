using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.ToyWorld.Control;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.IO;
using System.Windows.Forms.Design;
using Logger;
using GoodAI.ToyWorld.Language;
using ToyWorldFactory;
using YAXLib;

namespace GoodAI.ToyWorld
{
    public partial class ToyWorld : MyWorld, IMyVariableBranchViewNodeBase
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

        [MyOutputBlock(3), MyUnmanaged]
        public MyMemoryBlock<float> VisualTool
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> Text
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(5)]
        public MyMemoryBlock<float> ChosenActions
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
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


        #region Effects

        [MyBrowsable, Category("Effects - General"), DisplayName("Rotate Map")]
        [YAXSerializableField(DefaultValue = false)]
        public bool RotateMap { get; set; }


        [MyBrowsable, Category("Effects - Noise"), DisplayName("Draw noise")]
        [YAXSerializableField(DefaultValue = false)]
        public bool DrawNoise { get; set; }

        [MyBrowsable, Category("Effects - Noise"), DisplayName("Noise intensity")]
        [YAXSerializableField(DefaultValue = 0.5f)]
        public float NoiseIntensity { get; set; }


        [MyBrowsable, Category("Effects - Smoke"), DisplayName("Draw smoke")]
        [YAXSerializableField(DefaultValue = false)]
        public bool DrawSmoke { get; set; }

        [MyBrowsable, Category("Effects - Smoke"), DisplayName("Smoke intensity")]
        [YAXSerializableField(DefaultValue = 0.5f)]
        public float SmokeIntensity { get; set; }

        [MyBrowsable, Category("Effects - Smoke"), DisplayName("Smoke scale")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float SmokeScale { get; set; }

        [MyBrowsable, Category("Effects - Smoke"), DisplayName("Smoke transf. speed")]
        [YAXSerializableField(DefaultValue = 1.0f)]
        public float SmokeTransformationSpeed { get; set; }

        [MyBrowsable, Category("Effects - Lighting"), DisplayName("Day/Night cycle")]
        [YAXSerializableField(DefaultValue = false)]
        public bool EnableDayAndNightCycle { get; set; }

        [MyBrowsable, Category("Effects - Lighting"), DisplayName("Draw lights")]
        [YAXSerializableField(DefaultValue = false)]
        public bool DrawLights { get; set; }

        #endregion

        #region RenderRequests

        [MyBrowsable, Category("FoF view"), DisplayName("Size")]
        [YAXSerializableField(DefaultValue = 3)]
        public int FoFSize { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("Resolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoFResWidth { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("Resolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoFResHeight { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("Multisample level")]
        [YAXSerializableField(DefaultValue = RenderRequestMultisampleLevel.x4)]
        public RenderRequestMultisampleLevel FoFMultisampleLevel { get; set; }


        [MyBrowsable, Category("FoV view"), DisplayName("Size")]
        [YAXSerializableField(DefaultValue = 21)]
        public int FoVSize { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("Resolution width")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoVResWidth { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("Resolution height")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int FoVResHeight { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("Multisample level")]
        [YAXSerializableField(DefaultValue = RenderRequestMultisampleLevel.x4)]
        public RenderRequestMultisampleLevel FoVMultisampleLevel { get; set; }


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
        [YAXSerializableField(DefaultValue = RenderRequestMultisampleLevel.x4)]
        public RenderRequestMultisampleLevel FreeViewMultisampleLevel { get; set; }


        [MyBrowsable, Category("Tool display"), DisplayName("Size")]
        [YAXSerializableField(DefaultValue = 0.9f)]
        public float ToolSize { get; set; }

        [MyBrowsable, Category("Tool display"), DisplayName("Resolution width")]
        [YAXSerializableField(DefaultValue = 128)]
        public int ToolResWidth { get; set; }

        [MyBrowsable, Category("Tool display"), DisplayName("Resolution height")]
        [YAXSerializableField(DefaultValue = 128)]
        public int ToolResHeight { get; set; }

        [MyBrowsable, Category("Tool display"), DisplayName("Multisample level")]
        [YAXSerializableField(DefaultValue = RenderRequestMultisampleLevel.None)]
        public RenderRequestMultisampleLevel ToolMultisampleLevel { get; set; }

        #endregion


        [MyBrowsable, Category("Language interface"), DisplayName("Maximum message length")]
        [YAXSerializableField(DefaultValue = 128)]
        public int MaxMessageLength { get; set; }

        [MyBrowsable, Category("Language interface"), DisplayName("Word vector dimensions")]
        [YAXSerializableField(DefaultValue = 50)]
        public int WordVectorDimensions { get; set; }

        [MyBrowsable, Category("Language interface"), DisplayName("Maximum input words")]
        [YAXSerializableField(DefaultValue = 4)]
        public int MaxInputWordCount { get; set; }

        #endregion

        public Vocabulary Vocabulary { get; private set; }

        public IGameController GameCtrl { get; set; }
        public IAvatarController AvatarCtrl { get; set; }

        private IFovAvatarRR FovRR { get; set; }
        private IFofAvatarRR FofRR { get; set; }
        private IFreeMapRR FreeRR { get; set; }

        private int SignalCount { get; set; }

        private readonly Dictionary<string, int> m_controlIndexes = new Dictionary<string, int>();

        public ToyWorld()
        {
            if (TilesetTable == null)
                TilesetTable = GetDllDirectory() + @"\res\GameActors\Tiles\Tilesets\TilesetTable.csv";
            if (SaveFile == null)
                SaveFile = GetDllDirectory() + @"\res\Worlds\mockup999_pantry_world.tmx";

            SignalCount = GameFactory.GetSignalCount();
            AddOutputs(SignalCount, "Signal_");

            Vocabulary = new Vocabulary(WordVectorDimensions);
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

            if (Controls != null)
                validator.AssertError(Controls.Count >= 84 || Controls.Count == m_controlsCount, this, "Controls size has to be of size " + m_controlsCount + " or 84+. Use device input node for controls, or provide correct number of inputs");

            TryToyWorld();

            foreach (TWLogMessage message in TWLog.GetAllLogMessages())
                switch (message.Severity)
                {
                    case TWSeverity.Error:
                        {
                            validator.AssertError(false, this, message.ToString());
                            break;
                        }
                    case TWSeverity.Warn:
                        {
                            validator.AssertWarning(false, this, message.ToString());
                            break;
                        }
                }
        }

        private void TryToyWorld()
        {
            if (GameCtrl != null)
                GameCtrl.Dispose(); // Should dispose RRs and controllers too

            GameSetup setup = new GameSetup(
                    new FileStream(SaveFile, FileMode.Open, FileAccess.Read, FileShare.Read),
                    new StreamReader(TilesetTable));
            GameCtrl = GameFactory.GetThreadSafeGameController(setup);
            GameCtrl.Init();
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

            if (Controls == null)
                return;

            ChosenActions.Count = Controls.Count;

            if (Controls.Count == m_controlsCount)
            {
                MyLog.INFO.WriteLine("ToyWorld: Controls set to vector mode.");

                m_controlIndexes["forward"] = 0;
                m_controlIndexes["backward"] = 1;
                m_controlIndexes["left"] = 2;
                m_controlIndexes["right"] = 3;
                m_controlIndexes["rot_left"] = 4;
                m_controlIndexes["rot_right"] = 5;
                m_controlIndexes["fof_left"] = 6;
                m_controlIndexes["fof_right"] = 7;
                m_controlIndexes["fof_up"] = 8;
                m_controlIndexes["fof_down"] = 9;
                m_controlIndexes["interact"] = 10;
                m_controlIndexes["use"] = 11;
                m_controlIndexes["pickup"] = 12;
            }
            else if (Controls.Count >= 84)
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

        private void SetDummyOutputs(int howMany, string dummyName, int dummySize)
        {
            int idx = 1;
            for (int i = OutputBranches - howMany; i < OutputBranches; ++i)
            {
                MyMemoryBlock<float> mb = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
                mb.Name = dummyName + idx++;
                mb.Count = dummySize;
                m_outputs[i] = mb;
            }
        }

        private void AddOutputs(int branchesToAdd, string dummyName, int dummySize = 1)
        {
            int oldOutputBranches = OutputBranches;
            // backup current state of memory blocks -- setting value to OutputBranches will reset m_outputs
            List<MyAbstractMemoryBlock> backup = new List<MyAbstractMemoryBlock>();
            for (int i = 0; i < oldOutputBranches; ++i)
                backup.Add(m_outputs[i]);

            OutputBranches = oldOutputBranches + branchesToAdd;

            for (int i = 0; i < oldOutputBranches; ++i)
                m_outputs[i] = backup[i];

            SetDummyOutputs(SignalCount, dummyName, dummySize);
        }

        /// <summary>
        /// Returns Signal node with given index (from 0 to SignalCount)
        /// </summary>
        /// <param name="index">Index of Signal node</param>
        /// <returns></returns>
        public MyParentInput GetSignalNode(int index)
        {
            int offset = OutputBranches - SignalCount;
            return Owner.Network.GroupInputNodes[offset + index];
        }

        /// <summary>
        /// Returns memory block assigned to Signal with given index (from 0 to SignalCount)
        /// </summary>
        /// <param name="index">Index of Signal</param>
        /// <returns></returns>
        public MyMemoryBlock<float> GetSignalMemoryBlock(int index)
        {
            return GetSignalNode(index).Output;
        }
    }
}
