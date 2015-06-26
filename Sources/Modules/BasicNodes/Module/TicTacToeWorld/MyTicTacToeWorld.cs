using BrainSimulator.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrainSimulator.Utils;
using BrainSimulator.Memory;
using BrainSimulator.Task;
using System.ComponentModel;
using YAXLib;
using System.Drawing;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections;
using BrainSimulator.Signals;
using CustomModels.TicTacToeWorld;

namespace BrainSimulator.TicTacToe
{
    /// <author>Jaroslav Vitku</author>
    /// <status>Under Development</status>
    /// <summary>Simulator of Tic Tac Toe game.</summary>
    /// <description>
    /// 
    /// Player (the brain) has X and computer (world) has O. 
    /// Games are repeated. If the player wins, reward is received. If the computer wins, punishment is received.
    /// 
    /// <h3>Outputs:</h3>
    /// <ul>
    ///     <li> <b>Global:</b>Vectort of all state variables of the world  = [State, Reward, Punishment]</li>
    ///     <li> <b>State:</b>Vector of length 9 with values {0,1,2} = {empty, player, computer}.</li>
    ///     <li> <b>RewardEvent:</b> Reward is received if the player wins the game.</li>
    ///     <li> <b>PunishmentEvent:</b> Punishment is received if the player wins the game.</li>
    ///     <li> <b>Visual:</b> bitmap representing the current world state.</li>
    /// </ul>
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li> <b>Action:</b> Vector of size 9 which chosen action. Action = place, where to put the X. If X is already taken, the step is missed and the player has another attempt.</li>
    /// </ul>
    /// 
    /// </description>
    public class MyTicTacToeWorld : MyWorld
    {
        [MyInputBlock]
        public MyMemoryBlock<float> ActionInput
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Visual
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }
        
        [MyOutputBlock(1)]
        public MyMemoryBlock<float> StateOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> EventOutput
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 16)]
        public int RES { get; private set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 160)]
        public int VISIBLE_HEIGHT { get; private set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 160)]
        public int VISIBLE_WIDTH { get; private set; }

        public readonly int SX = 3;
        public readonly int SY = 3;
        public readonly int REWARD = 0;
        public readonly int PUNISHMENT = 1;

        private Dictionary<string, Bitmap> m_bitmapTable = new Dictionary<string, Bitmap>();
        public MyMemoryBlock<float> Bitmaps { get; private set; }
        private string m_errorMessage;

        private MyGraphicsPrototype m_tale_o, m_tale_empty, m_tale_x;
        protected MyGraphicsPrototype[] m_allGraphicsPrototypes;
        
        public class MyGraphicsPrototype
        {
            public int2 PixelSize;
            public CUdeviceptr Bitmap;
        }

        // definition of the map for GPU rendering, tales can be: empty/obstacle (empty ones may contain other objects)
        public MyMemoryBlock<int> MapTales { get; private set; }

        private string TEXTURE_SET = @"res\tictactoeworld\";
        protected int[] INIT_STATE;
        protected int[] m_currentState;

        public override void UpdateMemoryBlocks()
        {
            INIT_STATE= new int[SX * SY];
                
            Bitmaps.Count = 0;

            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleEmpty.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleO.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleX.png");

            MapTales.Count = SX * SY;
            MapTales.ColumnHint = SX;

            EventOutput.Count = 2;
            StateOutput.Count = SX * SY;

            VISIBLE_HEIGHT = SY * RES;
            VISIBLE_WIDTH = SX * RES;

            Visual.Count = SX * SY * RES * RES;
            Visual.ColumnHint = SX * RES;
        }

        // @see MyCustomPongWorld
        private int LoadAndGetBitmapSize(string path)
        {
            if (!m_bitmapTable.ContainsKey(path))
            {
                try
                {
                    Bitmap bitmap = (Bitmap)Image.FromFile(path, true);
                    m_bitmapTable[path] = bitmap;

                    return bitmap.Width * bitmap.Height;
                }
                catch (Exception ex)
                {
                    m_errorMessage = ex.Message;
                    return 0;
                }
            }
            else return m_bitmapTable[path].Width * m_bitmapTable[path].Height;
        }

        // @see MyCustomPongWorld
        private int FillWithChannelFromBitmap(Bitmap bitmap, int channel, float[] buffer, int offset)
        {
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

            byte[] pixels = new byte[bitmapData.Stride];

            int bytesPerPixel = bitmapData.Stride / bitmapData.Width;

            for (int i = 0; i < bitmap.Height; i++)
            {
                Marshal.Copy(bitmapData.Scan0, pixels, 0, pixels.Length);
                bitmapData.Scan0 += bitmapData.Stride;

                for (int j = 0; j < bitmap.Width; j++)
                {
                    buffer[i * bitmap.Width + j + offset] = pixels[j * bytesPerPixel + channel] / 255.0f;
                }
            }
            bitmap.UnlockBits(bitmapData);
            return bitmap.Width * bitmap.Height;
        }

        public override void Validate(MyValidator validator)
        {
        }

        public MyInitTask InitGameTask { get; private set; }
        public MyUpdateTask UpdateTask { get; private set; }
        public MyRenderTask RenderGameTask { get; private set; }

        /// <summary>
        /// Initialize the world (load graphics etc.).
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MyTicTacToeWorld>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                Owner.m_currentState = (int[])Owner.INIT_STATE.Clone();

                // load bitmaps and pas them to the game objects
                int offset = 0;
                CudaDeviceVariable<float> devBitmaps = Owner.Bitmaps.GetDevice(Owner);

                Bitmap bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "taleEmpty.png"];
                Owner.m_tale_empty = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "taleO.png"];
                Owner.m_tale_o = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "taleX.png"];
                Owner.m_tale_x = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                Owner.Bitmaps.SafeCopyToDevice();
                Array.Copy(Owner.INIT_STATE, Owner.MapTales.Host, Owner.MapTales.Count);
                Owner.MapTales.SafeCopyToDevice();

                Owner.m_allGraphicsPrototypes = new MyGraphicsPrototype[] { Owner.m_tale_empty, Owner.m_tale_o, Owner.m_tale_x };
            }


            // @see MyCustomPongWorld
            private int FillWithChannelFromBitmap(Bitmap bitmap, int channel, float[] buffer, int offset)
            {
                BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

                byte[] pixels = new byte[bitmapData.Stride];

                int bytesPerPixel = bitmapData.Stride / bitmapData.Width;

                for (int i = 0; i < bitmap.Height; i++)
                {
                    Marshal.Copy(bitmapData.Scan0, pixels, 0, pixels.Length);
                    bitmapData.Scan0 += bitmapData.Stride;

                    for (int j = 0; j < bitmap.Width; j++)
                    {
                        buffer[i * bitmap.Width + j + offset] = pixels[j * bytesPerPixel + channel] / 255.0f;
                    }
                }
                bitmap.UnlockBits(bitmapData);
                return bitmap.Width * bitmap.Height;
            }
        }

        /// <summary>
        /// Renders the visible area, not needed for simulation.
        /// </summary>
        public class MyRenderTask : MyTask<MyTicTacToeWorld>
        {
            private MyCudaKernel m_drawTalesKernel;
            private MyCudaKernel m_drawFreeObjKernel;

            public override void Init(int nGPU)
            {
                m_drawFreeObjKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\GridWorld", "DrawFreeObjectKernel");
                m_drawTalesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\GridWorld", "DrawTalesKernel");
            }

            private int2 ToPixelCoords(int pos)
            {
                int2 output = new int2(pos % Owner.SX, pos / Owner.SY);
                output.x = output.x * Owner.m_tale_empty.PixelSize.x;
                output.y = output.y * Owner.m_tale_empty.PixelSize.y + Owner.m_tale_empty.PixelSize.y - 1;
                return output;
            }

            private int2 ToPixelCoords(int2 pos)
            {
                int2 output;
                output.x = pos.x * Owner.m_tale_empty.PixelSize.x;
                output.y = pos.y * Owner.m_tale_empty.PixelSize.y + Owner.m_tale_empty.PixelSize.y - 1;
                return output;
            }

            public override void Execute()
            {
                // shares kernels with GridWorld, so the parameter taleObstacle is not needed
                MyGraphicsPrototype taleEmpty = Owner.m_tale_empty;
                MyGraphicsPrototype taleObstacle = Owner.m_tale_o;

                m_drawTalesKernel.SetupExecution(Owner.VISIBLE_WIDTH * Owner.VISIBLE_HEIGHT);
                m_drawTalesKernel.Run(Owner.Visual, Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT,
                    Owner.MapTales, Owner.SX, Owner.SY,
                    taleEmpty.Bitmap, taleObstacle.Bitmap, taleEmpty.PixelSize);

                MyGraphicsPrototype toDraw;
                int2 pos;

                for (int i = 0; i < Owner.m_currentState.Length; i++)
                {
                    pos = ToPixelCoords(i);
                    toDraw = Owner.m_allGraphicsPrototypes[Owner.m_currentState[i]];

                    m_drawFreeObjKernel.SetupExecution(toDraw.PixelSize.x * toDraw.PixelSize.y);
                    m_drawFreeObjKernel.Run(Owner.Visual, Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT, toDraw.Bitmap,
                            pos, toDraw.PixelSize);
                }
            }
        }


        /// <summary>
        /// Read action, if valid check for win, choose the computer action, check for win, 
        /// reset and publish reward/punishment if the game ends, publish new state.
        /// 
        /// <description>
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>Difficulty: </b>How well the computer plays</li>
        /// </ul>
        /// </description>
        /// </summary>
        public class MyUpdateTask : MyTask<MyTicTacToeWorld>
        {
            [MyBrowsable, Category("Configuration"),
            Description("How well the computer plays")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float Difficulty { get; set; }

            [MyBrowsable, Category("Configuration"),
            Description("Computer Startd")]
            [YAXSerializableField(DefaultValue = false)]
            public bool ComputerStarts{ get; set; }


            private MyTicTacToeGame game;
            private ITicTacToeEngine er;

            public override void Init(int nGPU)
            {
                Owner.m_currentState = (int[])Owner.INIT_STATE.Clone();

                game = new MyTicTacToeGame(Owner.m_currentState);
                //er = new MyEngineRandom();
                er = new MyEngineA(Difficulty, game);

                ResetGame();
            }

            private bool ge = false;

            public override void Execute()
            {
                PreSimulationStep();
                /*
                if (ge)
                {
                    ge = false;
                    ResetGame();
                    PostSimulationStep();
                    return;
                }*/

                int action = DecodeAction();

                if (!game.ApplyAction(action, MyTicTacToeGame.PLAYER, Owner.m_currentState))
                {
                    MyLog.DEBUG.WriteLine("Action already taken "+action);
                    PostSimulationStep();
                    return;
                }

                if (game.CheckWinner(MyTicTacToeGame.PLAYER, Owner.m_currentState))
                {
                    MyLog.INFO.WriteLine("PLAYER won");
                    ResetGame();
                    ge = true;
                    Owner.EventOutput.Host[Owner.REWARD] = 1;
                    PostSimulationStep();
                    return;
                }

                if (game.Ended(Owner.m_currentState))
                {
                    ge = true;
                    ResetGame();
                    PostSimulationStep();
                    return;
                }

                int ac = er.GenerateAction(Owner.m_currentState);
                while (!game.ApplyAction(ac, MyTicTacToeGame.COMPUTER, Owner.m_currentState))
                {
                    ac = er.GenerateAction(Owner.m_currentState);
                }

                if (game.CheckWinner(MyTicTacToeGame.COMPUTER, Owner.m_currentState))
                {
                    MyLog.INFO.WriteLine("COMPUTER won");
                    ge = true;
                    ResetGame();
                    Owner.EventOutput.Host[Owner.PUNISHMENT] = 1;
                    PostSimulationStep();
                    return;
                }

                if (game.Ended(Owner.m_currentState))
                {
                    ge = true;
                    ResetGame();
                    PostSimulationStep();
                    return;
                }
                PostSimulationStep();
            }

            private void PreSimulationStep()
            {
                ((MyEngineA)er).UpdateDifficulty(Difficulty);
                Owner.EventOutput.SafeCopyToHost();
                Owner.EventOutput.Host[Owner.REWARD] = 0;
                Owner.EventOutput.Host[Owner.PUNISHMENT] = 0;
                Owner.StateOutput.SafeCopyToHost();
            }

            private void PostSimulationStep()
            {
                Owner.EventOutput.SafeCopyToDevice();
                Array.Copy(Owner.m_currentState, Owner.StateOutput.Host, Owner.m_currentState.Length);
                Owner.StateOutput.SafeCopyToDevice();
            }

            private void ResetGame()
            {
                Owner.m_currentState = (int[]) Owner.INIT_STATE.Clone();
                if (ComputerStarts)
                {
                    int a = er.GenerateAction(Owner.m_currentState);
                    game.ApplyAction(a, MyTicTacToeGame.COMPUTER, Owner.m_currentState);
                }
            }

            // get action with the max (utility) value
            internal int DecodeAction()
            {
                if (Owner.ActionInput == null)
                {
                    return 0;
                }
                Owner.ActionInput.SafeCopyToHost();

                int result = 0;

                for (int i = 0; i < Owner.ActionInput.Host.Length; i++)
                {
                    if (Owner.ActionInput.Host[result] < Owner.ActionInput.Host[i])
                    {
                        result = i;
                    }
                }
                return result;
            }
        }
    }
}
