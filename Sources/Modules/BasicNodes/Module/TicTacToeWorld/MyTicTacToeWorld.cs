using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using YAXLib;

namespace GoodAI.Modules.TicTacToe
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>Simulator of Tic Tac Toe game.</summary>
    /// <description>
    /// 
    /// World for playing the TicTacToe game. The world has common output defining state of the game. 
    /// For each player there is a separate output with events (reward=won, punishment=lost, incorrect_move=place already taken) and input for actions (max value is taken).  
    /// 
    /// <h3>Outputs:</h3>
    /// <ul>
    ///     <li> <b>State:</b> Vector of length 9 with values {0,1,2} = {EMPTY, PLAYER_O, PLAYER_X}.</li>
    ///     <li> <b>EventO:</b> Vector of events for the PLAYER_O formatted as follows {REWARD, PUNISHMENT, INCORRECT_MOVE}.</li>
    ///     <li> <b>EventX:</b> Vector of events for the PLAYER_X formatted as follows {REWARD, PUNISHMENT, INCORRECT_MOVE}.</li>
    ///     <li> <b>Visual:</b> Bitmap representing the current game state.</li>
    /// </ul>
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li> <b>PlayerXAction:</b> Vector of size 9 which chosen action. Action = place, where to put the X (maximum value is taken as an action). If X is already taken, the step is missed and the player has another attempt.</li>
    ///     <li> <b>PlayerYAction:</b> Vector of size 9 which chosen action. Input is processed only if the corresponding signal is Raised.</li>
    /// </ul>
    /// 
    /// The world sends two signals which are supposed trigger players in conditional groups.
    /// </description>
    public class MyTicTacToeWorld : MyWorld
    {
        [MyInputBlock]
        public MyMemoryBlock<float> PlayerXActionInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> PlayerOActionInput
        {
            get { return GetInput(1); }
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
        public MyMemoryBlock<float> EventOOutput
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> EventXOutput
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
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


        public static readonly int NO_POSITIONS = 9;
        public enum PLAYERS
        {
            PLAYER_O = 1,
            PLAYER_X = 2
        };

        public enum TALES
        {
            EMPTY = 0,
            PLAYER_O = 1,
            PLAYER_X = 2
        }

        public readonly int SX = 3;
        public readonly int SY = 3;
        public readonly int REWARD_POS = 0;
        public readonly int PUNISHMENT_POS = 1;
        public readonly int INCORRECT_MOVE_POS = 2;

        public class MyPlayerXSignal : MySignal { } // plays in single player mode
        public class MyPlayerOSignal : MySignal { }

        public MyPlayerXSignal AgentXPlays { get; private set; }
        public MyPlayerOSignal AgentOPlays { get; private set; }

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

        public MyInitTask InitGameTask { get; private set; }
        public MyMultiPlayerTask MultiPlayer { get; private set; }
        public MyRenderTask RenderGameTask { get; private set; }

        private string TEXTURE_SET = @"res\tictactoeworld\";
        protected int[] INIT_STATE;
        protected int[] m_currentState;

        private MyTicTacToeGame game;
        private Random r;

        public override void UpdateMemoryBlocks()
        {
            Bitmaps.Count = 0;
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleEmpty.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleO.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleX.png");

            MapTales.Count = SX * SY;
            MapTales.ColumnHint = SX;

            EventOOutput.Count = 3;
            EventXOutput.Count = 3;
            StateOutput.Count = SX * SY;

            VISIBLE_HEIGHT = SY * RES;
            VISIBLE_WIDTH = SX * RES;

            Visual.Count = SX * SY * RES * RES;
            Visual.ColumnHint = SX * RES;

            INIT_STATE = new int[SX * SY];
            m_currentState = (int[])INIT_STATE.Clone();
            game = new MyTicTacToeGame(m_currentState);
            r = new Random();
        }

        #region BitmapProcessing

        // @see MyCustomPongWorld
        private int LoadAndGetBitmapSize(string path)
        {
            if (!m_bitmapTable.ContainsKey(path))
            {
                try
                {
                    Bitmap bitmap = (Bitmap)Image.FromFile(MyResources.GetMyAssemblyPath() + "\\" + path, true);
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

        #endregion

        #region Drawing
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
        #endregion


        /// <summary>
        /// Read action of a given player. If the aciton is incorrect, sets the event and the same player has another attempt.
        /// 
        /// If the action is valid, signal for another player is raised.
        /// 
        /// If some of players won, the corresponding events are set and game ends (after both players have seen the event).
        /// 
        /// <description>
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>Player O Starts: </b>Whether the Player_O should start.</li>
        ///     <li><b>Randomize who starts: </b>Randomize who has the first move?</li>
        /// </ul>
        /// </description>
        /// </summary>
        public class MyMultiPlayerTask : MyTask<MyTicTacToeWorld>
        {
            [MyBrowsable, Category("Configuration"),
            Description("Player O Starts")]
            [YAXSerializableField(DefaultValue = false)]
            public bool PlayerXStarts { get; set; }

            [MyBrowsable, Category("Configuration"),
            Description("Randomize who starts")]
            [YAXSerializableField(DefaultValue = true)]
            public bool RandomizeStartingPlayer { get; set; }

            private PLAYERS m_justPlayed, m_justNotPlayed;
            private int m_stepsToEnd = -1;

            public override void Init(int nGPU)
            {
                m_stepsToEnd = -1;
                ResetGame();
            }

            private void PreSimulationStep()
            {
                Owner.EventOOutput.SafeCopyToHost();
                Owner.EventXOutput.SafeCopyToHost();
                Owner.EventOOutput.Host[Owner.INCORRECT_MOVE_POS] = 0;
                Owner.EventXOutput.Host[Owner.INCORRECT_MOVE_POS] = 0;
            }

            private void PostSimulationStep(bool shouldSwitchPlayers)
            {
                // publish new state
                Owner.StateOutput.SafeCopyToHost();
                Array.Copy(Owner.m_currentState, Owner.StateOutput.Host, Owner.m_currentState.Length);
                Owner.StateOutput.SafeCopyToDevice();

                if (shouldSwitchPlayers)
                    SwitchPlayers();

                Owner.EventOOutput.SafeCopyToDevice();
                Owner.EventXOutput.SafeCopyToDevice();
            }

            public override void Execute()
            {
                PreSimulationStep();

                // signal that the game ended, just to publish signals to both players
                if (m_stepsToEnd >= 0)
                {
                    // all signals sent, new game
                    if (m_stepsToEnd == 0)
                    {
                        ResetGame();
                        PostSimulationStep(true);
                        return;
                    }
                    m_stepsToEnd--;
                    PostSimulationStep(true);
                    return;
                }

                int action = DecodeAction();

                if (!Owner.game.ApplyAction(action, m_justPlayed, Owner.m_currentState))
                {
                    Owner.GetOutput((int)m_justPlayed + 1).Host[Owner.INCORRECT_MOVE_POS] = 1;
                    PostSimulationStep(false);
                    return;
                }
                if (Owner.game.CheckWinner(m_justPlayed, Owner.m_currentState))
                {
                    m_stepsToEnd = 1;
                    Owner.GetOutput((int)m_justPlayed + 1).Host[Owner.REWARD_POS] = 1;
                    Owner.GetOutput((int)m_justNotPlayed + 1).Host[Owner.PUNISHMENT_POS] = 1;
                    PostSimulationStep(true);
                    return;
                }

                if (MyTicTacToeGame.NoFreePlace(Owner.m_currentState))
                {
                    m_stepsToEnd = 0;
                }

                PostSimulationStep(true);
            }

            private void RememberJustPlayed(PLAYERS played)
            {
                if (played == PLAYERS.PLAYER_X)
                {
                    m_justPlayed = PLAYERS.PLAYER_X;
                    m_justNotPlayed = PLAYERS.PLAYER_O;
                }
                else
                {
                    m_justPlayed = PLAYERS.PLAYER_O;
                    m_justNotPlayed = PLAYERS.PLAYER_X;
                }
            }

            private int DecodeAction()
            {
                int inputNo;
                if (m_justPlayed == PLAYERS.PLAYER_X)
                {
                    inputNo = 0;
                }
                else
                {
                    inputNo = 1;
                }

                if (Owner.GetInput(inputNo) == null)
                {
                    return 0;
                }
                Owner.GetInput(inputNo).SafeCopyToHost();

                int result = 0;

                for (int i = 0; i < Owner.GetInput(inputNo).Host.Length; i++)
                {
                    if (Owner.GetInput(inputNo).Host[result] < Owner.GetInput(inputNo).Host[i])
                    {
                        result = i;
                    }
                }
                return result;
            }

            private void ResetGame()
            {
                Owner.m_currentState = (int[])Owner.INIT_STATE.Clone();

                m_stepsToEnd = -1;
                CleanEvents();

                bool playerXStarts = PlayerXStarts;
                if (RandomizeStartingPlayer)
                {
                    playerXStarts = Owner.r.NextDouble() >= 0.5;
                }
                SetXPlays(playerXStarts);
            }

            private void CleanEvents()
            {
                Owner.EventOOutput.SafeCopyToHost();
                Owner.EventXOutput.SafeCopyToHost();

                for (int i = 0; i < Owner.EventOOutput.Count; i++)
                {
                    Owner.EventOOutput.Host[i] = 0;
                    Owner.EventXOutput.Host[i] = 0;
                }
                Owner.EventOOutput.SafeCopyToDevice();
                Owner.EventXOutput.SafeCopyToDevice();
            }

            private void SwitchPlayers()
            {
                SetXPlays(!Owner.AgentXPlays.IsRised());
            }


            private void SetXPlays(bool playerX)
            {
                if (playerX)
                {
                    RememberJustPlayed(PLAYERS.PLAYER_X);
                    Owner.AgentXPlays.Raise();
                    Owner.AgentOPlays.Drop();
                }
                else
                {
                    RememberJustPlayed(PLAYERS.PLAYER_O);
                    Owner.AgentXPlays.Drop();
                    Owner.AgentOPlays.Raise();
                }
            }
        }
    }
}
