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

namespace GoodAI.ToyWorld
{
    public class ToyWorld : MyWorld
    {
        public TWUpdateTask UpdateTask { get; private set; }

        public TWGetInputTask GetInputTask { get; private set; }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> VisualFov
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> VisualFof
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> VisualFree
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Controls
        {
            get { return GetInput(0); }
        }

        [MyBrowsable, Category("Runtime"), DisplayName("Run every Nth")]
        [YAXSerializableField(DefaultValue = 1)]
        public int RunEvery { get; set; }

        [MyBrowsable, Category("Files"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        [YAXSerializableField(DefaultValue = null), YAXCustomSerializer(typeof(MyPathSerializer))]
        public string TilesetTable { get; set; }

        [MyBrowsable, Category("Files"), EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        [YAXSerializableField(DefaultValue = null), YAXCustomSerializer(typeof(MyPathSerializer))]
        public string SaveFile { get; set; }

        [MyBrowsable, Category("FoF view"), DisplayName("FoF size")]
        [YAXSerializableField(DefaultValue = 3)]
        public int FoFSize { get; set; }

        [MyBrowsable, Category("FoV view"), DisplayName("FoV size")]
        [YAXSerializableField(DefaultValue = 21)]
        public int FoVSize { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tCenter - X")]
        [YAXSerializableField(DefaultValue = 0)]
        public float CenterX { get; set; }

        [MyBrowsable, Category("Free view"), DisplayName("\tCenter - Y")]
        [YAXSerializableField(DefaultValue = 0)]
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

        private IGameController m_gameCtrl { get; set; }

        private IAvatarController m_avatarCtrl { get; set; }

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
            validator.AssertError(Width > 0, this, "Free view width has to be positive.");
            validator.AssertError(Height > 0, this, "Free view height has to be positive.");
            validator.AssertError(ResolutionWidth > 0, this, "Free view resolution width has to be positive.");
            validator.AssertError(ResolutionHeight > 0, this, "Free view resolution height has to be positive.");

            if (Controls != null)
                validator.AssertError(Controls.Count >= 84 || Controls.Count == 13, this, "Controls size has to be of size 13 or 84+. Use device input node for controls, or provide correct number of inputs");
        }

        public override void UpdateMemoryBlocks()
        {
            if (!File.Exists(SaveFile) || !File.Exists(TilesetTable) || FoFSize <= 0 || Width <= 0 || Height <= 0 || ResolutionWidth <= 0 || ResolutionHeight <= 0)
                return;

            GameSetup setup = new GameSetup(new FileStream(SaveFile, FileMode.Open, FileAccess.Read, FileShare.Read), new StreamReader(TilesetTable));
            m_gameCtrl = GameFactory.GetThreadSafeGameController(setup);
            m_gameCtrl.Init();

            int[] avatarIds = m_gameCtrl.GetAvatarIds();
            if (avatarIds.Length == 0)
            {
                MyLog.ERROR.WriteLine("No avatar found in map!");
                return;
            }

            int myAvatarId = avatarIds[0];
            m_avatarCtrl = m_gameCtrl.GetAvatarController(myAvatarId);

            m_fovRR = ObtainRR<IFovAvatarRR>(VisualFov, myAvatarId, (IRenderRequestBase rr) => { rr.Size = new SizeF(FoVSize, FoVSize); });
            m_fofRR = ObtainRR<IFofAvatarRR>(VisualFof, myAvatarId, (IRenderRequestBase rr) => { (rr as IFofAvatarRR).FovAvatarRenderRequest = m_fovRR; rr.Size = new SizeF(FoFSize, FoFSize); });
            m_freeRR = ObtainRR<IFreeMapRR>(VisualFree, (IRenderRequestBase rr) => { rr.Size = new SizeF(Width, Height); rr.Resolution = new Size(ResolutionWidth, ResolutionHeight); });
            m_freeRR.SetPositionCenter(CenterX, CenterY);
        }

        private static string GetDllDirectory()
        {
            return Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        }

        private T InitRR<T>(IRenderRequestBase rr, MyMemoryBlock<float> targetMemBlock, Action<IRenderRequestBase> initializer = null) where T : class, IRenderRequestBase
        {
            rr.GatherImage = true;
            if (initializer != null)
                initializer.Invoke(rr);

            targetMemBlock.Count = rr.Resolution.Width * rr.Resolution.Height;
            return rr as T;
        }

        private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, int avatarId, Action<IRenderRequestBase> initializer = null) where T : class, IAvatarRenderRequest
        {
            IRenderRequestBase rr = m_gameCtrl.RegisterRenderRequest<T>(avatarId);
            return InitRR<T>(rr, targetMemBlock, initializer);
        }

        private T ObtainRR<T>(MyMemoryBlock<float> targetMemBlock, Action<IRenderRequestBase> initializer = null) where T : class, IRenderRequest
        {
            IRenderRequestBase rr = m_gameCtrl.RegisterRenderRequest<T>();
            return InitRR<T>(rr, targetMemBlock, initializer);
        }

        public class TWGetInputTask : MyTask<ToyWorld>
        {
            private Dictionary<string, int> controlIndexes = new Dictionary<string, int>();

            public override void Init(int nGPU)
            {
                if (Owner.Controls.Count == 13)
                {
                    MyLog.INFO.WriteLine("ToyWorld: Controls set to WSAD mode.");
                    controlIndexes["forward"] = 0;
                    controlIndexes["backward"] = 1;
                    controlIndexes["left"] = 2;
                    controlIndexes["right"] = 3;
                    controlIndexes["rot_left"] = 4;
                    controlIndexes["rot_right"] = 5;
                    controlIndexes["fof_right"] = 6;
                    controlIndexes["fof_left"] = 7;
                    controlIndexes["fof_up"] = 8;
                    controlIndexes["fof_down"] = 9;
                    controlIndexes["interact"] = 10;
                    controlIndexes["use"] = 11;
                    controlIndexes["pickup"] = 12;
                }
                else if (Owner.Controls.Count >= 84)
                {
                    MyLog.INFO.WriteLine("ToyWorld: Controls set to keyboard mode.");
                    controlIndexes["forward"] = 87;     // W
                    controlIndexes["backward"] = 83;    // S
                    controlIndexes["rot_left"] = 65;        // A
                    controlIndexes["rot_right"] = 68;       // D
                    controlIndexes["left"] = 81;    // Q
                    controlIndexes["right"] = 69;   // E

                    controlIndexes["fof_up"] = 73;      // I
                    controlIndexes["fof_left"] = 76;    // J
                    controlIndexes["fof_down"] = 75;    // K
                    controlIndexes["fof_right"] = 74;   // L

                    controlIndexes["interact"] = 66;    // B
                    controlIndexes["use"] = 78;         // N
                    controlIndexes["pickup"] = 77;      // M
                }
            }

            public override void Execute()
            {
                if (SimulationStep != 0 && SimulationStep % Owner.RunEvery != 0)
                    return;

                Owner.Controls.SafeCopyToHost();
                float leftSignal = Owner.Controls.Host[controlIndexes["left"]];
                float rightSignal = Owner.Controls.Host[controlIndexes["right"]];
                float fwSignal = Owner.Controls.Host[controlIndexes["forward"]];
                float bwSignal = Owner.Controls.Host[controlIndexes["backward"]];
                float rotLeftSignal = Owner.Controls.Host[controlIndexes["rot_left"]];
                float rotRightSignal = Owner.Controls.Host[controlIndexes["rot_right"]];

                float fof_left = Owner.Controls.Host[controlIndexes["fof_left"]];
                float fof_right = Owner.Controls.Host[controlIndexes["fof_right"]];
                float fof_up = Owner.Controls.Host[controlIndexes["fof_up"]];
                float fof_down = Owner.Controls.Host[controlIndexes["fof_down"]];

                float rotation = convertBiControlToUniControl(rotLeftSignal, rotRightSignal);
                float speed = convertBiControlToUniControl(fwSignal, bwSignal);
                float rightSpeed = convertBiControlToUniControl(leftSignal, rightSignal);
                float fof_x = convertBiControlToUniControl(fof_left, fof_right);
                float fof_y = convertBiControlToUniControl(fof_up, fof_down);

                bool interact = Owner.Controls.Host[controlIndexes["interact"]] > 0.5 ? true : false;
                bool use = Owner.Controls.Host[controlIndexes["use"]] > 0.5 ? true : false;
                bool pickup = Owner.Controls.Host[controlIndexes["pickup"]] > 0.5 ? true : false;

                IAvatarControls ctrl = new AvatarControls(100, speed, rightSpeed, rotation, interact, use, pickup, fof: new PointF(fof_x, fof_y));
                Owner.m_avatarCtrl.SetActions(ctrl);
            }

            private float convertBiControlToUniControl(float a, float b)
            {
                return a >= b ? a : -b;
            }
        }

        public class TWUpdateTask : MyTask<ToyWorld>
        {
            public override void Init(int nGPU) { }

            private void PrintLogMessage(MyLog logger, TWLogMessage message)
            {
                logger.WriteLine("TWLog: " + message);
            }

            private void PrintLogMessages()
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
                        case TWSeverity.Verbose:
                        case TWSeverity.Debug:
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

                Owner.m_gameCtrl.MakeStep();

                TransferFromRRToMemBlock(Owner.m_fovRR, Owner.VisualFov);
                TransferFromRRToMemBlock(Owner.m_fofRR, Owner.VisualFof);
                TransferFromRRToMemBlock(Owner.m_freeRR, Owner.VisualFree);
            }

            private void TransferFromRRToMemBlock(IRenderRequestBase rr, MyMemoryBlock<float> mb)
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
