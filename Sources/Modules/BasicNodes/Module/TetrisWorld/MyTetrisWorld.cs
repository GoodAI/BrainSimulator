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
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using YAXLib;

namespace GoodAI.Modules.TetrisWorld
{
    /// <author>GoodAI</author>
    /// <meta>mp</meta>
    /// <status>Working</status>
    /// <summary> World for the Tetris game.</summary>
    /// <description>
    /// <p>
    /// World simulating <a href="https://en.wikipedia.org/wiki/Tetris">Tetris</a>. <br />
    /// The area where tetrominos fall is 10 cells wide and 22 cells high, with the top two cells obscured.<br />
    /// The tetrominos are spawned randomly at the top. When a row is cleared, all the rows above it are shifted down. <br />
    /// The score is calculated as &#931;(lines cleared<sub>t</sub>)<sup>2</sup> for all time steps t. <br />
    /// Every time an n-th row is cleared, the game advances the level. By default, n is 50. 
    /// The levels differ in the speed of the falling tetrominos. <br />
    /// For each successive level, the tetrominos wait 1 step less before falling down one cell.<br />
    /// The game is controlled via left, right, rotate left, rotate right and down actions.
    /// </p>
    /// 
    /// <p>
    /// The agent can, at any timestep:
    /// <ul>
    ///     <li>Output the action he wants to make. The action can be given to the world either as a vector or 
    ///     as an index into the table of actions, depending on the ActionInputModality parameter.</li>
    /// </ul>
    /// </p>
    /// 
    /// The world is composed of:
    /// <ul>
    ///     <li>An area where tetrominos fall.</li>
    ///     <li>A next tetromino hint.</li>
    ///     <li>A score counter.</li>
    ///     <li>A level indication.</li>
    /// </ul>
    /// 
    /// The world resets when the agent tops out, i.e. the spawned tetromino overlaps a previously placed tetromino.
    /// 
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li><b>ClearedLinesPerLevel:</b> number of cleared lines required to increment the game's level (and speed). 
    ///     It is 50 by default.</li>
    ///     <li><b>AlmostFullLinesAtStart:</b> number of almost full rows that the game begins with. 
    ///     Keeping this parameter non-zero may help the training.</li>
    ///     <li><b>WaitStepsPerFall:</b> number of timesteps the tetromino waits before moving down by one cell.</li>
    /// </ul>
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li><b>ActionInput:</b> the action applied upon the falling tetromino in the next time step. The possible actions are:
    ///     No action (0), Move left (1), Move right (2), Move down (3), Rotate left (4), Rotate right (5).</li>
    /// </ul>
    /// 
    /// <h3>Outputs</h3>
    /// <ul>
    ///     <li><b>BrickAreaOutput:</b> the area where bricks fall (10x22), with empty cell represented as 0 and the rest with numbers (1-7).</li>
    ///     <li><b>HintAreaOutput:</b> the area where the next brick is indicated (6x4), together with the indication of the next brick.</li>
    ///     <li><b>NextBrickNumberOutput:</b> A number (1-7) representing the next brick.</li>
    ///     <li><b>ScoreOutput:</b> the game's score, as a single number.</li>
    ///     <li><b>ScoreDeltaOutput:</b> the increase of the game's score from the previous time step, useful as a reward.</li>
    ///     <li><b>LevelOutput:</b> current level, starting at 0.</li>
    ///     <li><b>WorldEventOutput:</b> a number that indicates different world events that occur at each time step: 
    ///         0 = no event, 1 = lines were cleared, -1 = game over + reset</li>
    ///     <li><b>VisualOutput:</b> a bitmap that represents the complete game board.</li>
    /// </ul>
    /// 
    /// </description>
    public class TetrisWorld : MyWorld
    {
        // inspired by MyMastermindWorld
        // implementation note: the world is tightly coupled with its tasks and with the engine.

        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> ActionInput
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> BrickAreaOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> HintAreaOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> NextBrickNumberOutput
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> ScoreOutput
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> ScoreDeltaOutput
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        [MyOutputBlock(5)]
        public MyMemoryBlock<float> LevelOutput
        {
            get { return GetOutput(5); }
            set { SetOutput(5, value); }
        }

        [MyOutputBlock(6)]
        public MyMemoryBlock<float> WorldEventOutput
        {
            get { return GetOutput(6); }
            set { SetOutput(6, value); }
        }

        [MyOutputBlock(7)]
        public MyMemoryBlock<float> VisualOutput
        {
            get { return GetOutput(7); }
            set { SetOutput(7, value); }
        }

        public MyMemoryBlock<float> Bitmaps { get; protected set; }

        #endregion

        private Dictionary<TetrisWorld.TextureType, Bitmap> m_bitmapTable = new Dictionary<TetrisWorld.TextureType, Bitmap>();
        public TetrisWorldEngine Engine;
        public WorldEngineParams EngineParams = new WorldEngineParams();

        #region Parameters

        // The input is either a single number corresponding to a ActionInputType value or a vector.
        // If it's a vector, its length equals the cardinality of ActionInputType and the input action
        // is the index of the maximum value in the vector. This allows you to present a one-hot vector like
        //      0 0 1 0 0 0 0
        // or a vector with confidence values / probabilities like
        //      0.1 0 0.6 0.2 0 0.1
        // In the examples above, the third value (MoveRight == 2) of ActionInputType will be used.
        public enum InputModality
        {
            Vector = 0,
            Number = 1
        }

        [MyBrowsable, Category("I/O"), Description("ActionInput as a vector or as a single number.")]
        [YAXSerializableField(DefaultValue = InputModality.Vector)]
        public InputModality ActionInputModality { get; set; }

        [MyBrowsable, Category("Difficulty"), Description("Number of cleared lines before level is incremented.")]
        [YAXSerializableField(DefaultValue = 50)]
        public int ClearedLinesPerLevel {
            get { return EngineParams.ClearedLinesPerLevel; }
            set { EngineParams.ClearedLinesPerLevel = value; }
        }

        [MyBrowsable, Category("Difficulty"), Description("Number of steps before a tetromino falls down 1 cell.")]
        [YAXSerializableField(DefaultValue = 5)]
        public int WaitStepsPerFall
        {
            get { return EngineParams.WaitStepsPerFall; }
            set { EngineParams.WaitStepsPerFall = value; }
        }

        [MyBrowsable, Category("Difficulty"), Description("Number of almost full lines present at the start of the game.")]
        [YAXSerializableField(DefaultValue = 5)]
        public int AlmostFullLinesAtStart
        {
            get { return EngineParams.AlmostFullLinesAtStart; }
            set { EngineParams.AlmostFullLinesAtStart = value; }
        }

        [Category("Constants")]
        [YAXSerializableField(DefaultValue = 100)]
        public int VisualWidth { get; protected set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 100)]
        public int VisualHeight { get; protected set; }

        #endregion

        #region Graphics parameters

        public int BrickAreaColumns { get { return 10; } }
        public int BrickAreaRows { get { return 22; } }
        public int HintAreaColumns { get { return 4; } }
        public int HintAreaRows { get { return 4; } }

        private string textureSet = @"res\tetrisworld\";
        
        private const int GFX_CELL_SQUARE_WIDTH = 30;
        private Point GFX_BRICK_AREA_TOP_LEFT = new Point(4, 4);
        private Point GFX_HINT_AREA_TOP_LEFT = new Point(321, 94);
        private Point GFX_TEXT_AREA_TOP_LEFT = new Point(317, 5);
        
        protected const int SOURCE_VALUES_PER_PIXEL = 4; // RGBA: 4 floats per pixel
        protected const int TARGET_VALUES_PER_PIXEL = 3; // RGB: 3 floats per pixel

        #endregion

        /// <summary>
        /// Actions that the agent can submit to the world's ActionInput
        /// </summary>
        public enum ActionInputType
        {
            NoAction = 0,
            MoveLeft = 1,
            MoveRight = 2,
            MoveDown = 3,
            RotateLeft = 4,
            RotateRight = 5
        }

        /// <summary>
        /// Returns color associated to the specified brick type
        /// </summary>
        /// <param name="brick"></param>
        /// <returns></returns>
        public Color GetBrickColor(BrickType brick)
        {
            switch(brick)
            {
                case BrickType.I: return Color.Cyan;
                case BrickType.J: return Color.Blue;
                case BrickType.L: return Color.Orange;
                case BrickType.O: return Color.Yellow;
                case BrickType.S: return Color.Lime;
                case BrickType.T: return Color.Purple;
                case BrickType.Z: return Color.Red;
                case BrickType.Preset: return Color.Gray;
                case BrickType.None:
                default:
                    return Color.Black;
            }
        }

        /// <summary>
        /// Textures used for painting VisualOutput
        /// </summary>
        public enum TextureType
        {
            Background = 0,
            BrickOverlay = 1,
            BrickMask = 2,
            TextArea = 3
        }

        public TetrisWorld()
        {
            LoadBitmap(TextureType.Background, textureSet + "tetris_background.png");
            LoadBitmap(TextureType.BrickOverlay, textureSet + "brick_overlay.png");
            LoadBitmap(TextureType.BrickMask, textureSet + "brick_mask.png");
            LoadBitmap(TextureType.TextArea, textureSet + "text_area.png");
        }

        public override void UpdateMemoryBlocks()
        {
            Bitmaps.Count = 0;
            Bitmaps.Count += GetBitmapSize(TextureType.Background);
            Bitmaps.Count += GetBitmapSize(TextureType.BrickOverlay);
            Bitmaps.Count += GetBitmapSize(TextureType.BrickMask);
            Bitmaps.Count += GetBitmapSize(TextureType.TextArea);

            BrickAreaOutput.Count = BrickAreaColumns * BrickAreaRows;
            BrickAreaOutput.ColumnHint = BrickAreaColumns;

            HintAreaOutput.Count = HintAreaColumns * HintAreaRows;
            HintAreaOutput.ColumnHint = HintAreaColumns;

            NextBrickNumberOutput.Count = 1;

            ScoreOutput.Count = 1;

            ScoreDeltaOutput.Count = 1;

            LevelOutput.Count = 1;

            WorldEventOutput.Count = 1;

            VisualWidth = m_bitmapTable[TextureType.Background].Width;
            VisualHeight = m_bitmapTable[TextureType.Background].Height;
            VisualOutput.Count = VisualWidth * VisualHeight * TARGET_VALUES_PER_PIXEL;
            VisualOutput.ColumnHint = VisualWidth;
        }      

        /// <summary>
        /// Loads a bitmap from a file and stores it in a dictionary. Checks for ARGB color format (e.g. 32bit png). 
        /// Implementation same as in MyMastermindWorld (TODO: refactor once TetrisWorld gets moved to BasicNodes).
        /// </summary>
        /// <param name="path"></param>
        /// <param name="textureType"></param>
        protected void LoadBitmap(TextureType textureType, string path)
        {
            if (!m_bitmapTable.ContainsKey(textureType))
            {
                try
                {
                    Bitmap bitmap = (Bitmap)Image.FromFile(MyResources.GetMyAssemblyPath() + "\\" + path, true);
                    m_bitmapTable[textureType] = bitmap;

                    if(bitmap.PixelFormat != PixelFormat.Format32bppArgb)
                    {
                        throw new ArgumentException("The specified image is not in the required RGBA format."); // note: alpha must not be premultiplied
                    }
                }
                catch (Exception ex)
                {
                    MyLog.WARNING.WriteLine(ex.Message);
                }
            }
        }

        /// <summary>
        /// Returns the number of floats needed to represent the bitmap in a kernel.
        /// Implementation same as in MyMastermindWorld.
        /// </summary>
        /// <param name="textureType"></param>
        /// <returns></returns>
        protected int GetBitmapSize(TetrisWorld.TextureType textureType)
        {
            if (!m_bitmapTable.ContainsKey(textureType))
            {
                MyLog.WARNING.WriteLine("No bitmap was loaded for texture {0}", textureType.ToString());
                return 0;
            }
            else return m_bitmapTable[textureType].Width * m_bitmapTable[textureType].Height * SOURCE_VALUES_PER_PIXEL;
        }

        /// <summary>
        /// Fills the buffer with data from bitmap. The whole image data for R component is first, then G, then B, then A.
        /// Implementation same as in MyMastermindWorld.
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="buffer"></param>
        /// <param name="offset"></param>
        /// <returns>The number of floats needed to store bitmap in buffer</returns>
        public static int FillWithChannelsFromBitmap(Bitmap bitmap, float[] buffer, int offset)
        {
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

            byte[] pixels = new byte[bitmapData.Stride];
            int bytesPerPixel = bitmapData.Stride / bitmapData.Width;
            Debug.Assert(bytesPerPixel == 4); // we expect a 32-bit ARGB bitmap

            int totalPixels = bitmapData.Width * bitmapData.Height;
            int rOffset = 0;
            int gOffset = totalPixels;
            int bOffset = 2 * totalPixels;
            int aOffset = 3 * totalPixels;

            for (int i = 0; i < bitmap.Height; i++)
            {
                Marshal.Copy(bitmapData.Scan0, pixels, 0, pixels.Length);
                bitmapData.Scan0 += bitmapData.Stride;

                for (int j = 0; j < bitmap.Width; j++)
                {
                    int pixelIndex = (i * bitmap.Width + j) + offset;
                    // RGBA:
                    buffer[pixelIndex + rOffset] = pixels[j * bytesPerPixel + 0] / 255.0f; // R
                    buffer[pixelIndex + gOffset] = pixels[j * bytesPerPixel + 1] / 255.0f; // G
                    buffer[pixelIndex + bOffset] = pixels[j * bytesPerPixel + 2] / 255.0f; // B
                    buffer[pixelIndex + aOffset] = pixels[j * bytesPerPixel + 3] / 255.0f; // A
                }
            }
            bitmap.UnlockBits(bitmapData);
            return bitmap.Width * bitmap.Height * SOURCE_VALUES_PER_PIXEL;
        }

        private void ValidateRange(MyValidator validator, int value, int valueMin, int valueMax, string valueName)
        {
            if(value > valueMax)
            {
                validator.AddError(this, string.Format("{0} cannot be more than {1}.", valueName, valueMax));
            }
            if(value < valueMin)
            {
                validator.AddError(this, string.Format("{0} cannot be less than {1}.", valueName, valueMin));
            }
        }

        public override void Validate(MyValidator validator)
        {
            ValidateRange(validator, ClearedLinesPerLevel, 1, 1000000000, "ClearedLinesPerLevel");
            ValidateRange(validator, AlmostFullLinesAtStart, 0, 15, "AlmostFullLinesAtStart");
            if(ActionInput != null)
            {
                // make sure ActionInput has the required modality
                if(ActionInputModality == InputModality.Number && ActionInput.Count != 1)
                {
                    validator.AddError(this, "InputModality is Number. A single number is expected on ActionInput.");
                }
                else if(ActionInputModality == InputModality.Vector && ActionInput.Count != 6)
                {
                    validator.AddError(this, "InputModality is Vector. A vector of length 6 is expected on ActionInput.");
                }
            }
        }

        public InitTask InitGameTask { get; protected set; }
        public UpdateTask UpdateGameTask { get; protected set; }
        public RenderTask RenderGameTask { get; protected set; }

        // based on MyMastermindWorld.MyCudaTexture (TODO: refactor once TetrisWorld is in BasicNodes)
        public class CudaTexture 
        {
            public int2 SizeInPixels;
            public CUdeviceptr BitmapPtr;
            public int DataOffset;
        }

        protected CudaTexture m_textureBackground = new CudaTexture(),
            m_textureBrickOverlay = new CudaTexture(),
            m_textureBrickMask = new CudaTexture(),
            m_textureText = new CudaTexture();

        /// <summary>
        /// Initialize the world (load graphics, create engine).
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class InitTask : MyTask<TetrisWorld>
        {
            public override void Init(int nGPU)
            {
                Owner.Engine = new TetrisWorldEngine(Owner, Owner.EngineParams);
            }

            public override void Execute()
            {
                Owner.Engine.Reset();

                // load bitmaps to Bitmaps memory block; remembers their offsets in Owner's texture variables
                CudaDeviceVariable<float> devBitmaps = Owner.Bitmaps.GetDevice(Owner);
                TextureType[] textureTypes = { TextureType.Background, TextureType.BrickOverlay, 
                                             TextureType.BrickMask, TextureType.TextArea};
                CudaTexture[] textures = { Owner.m_textureBackground, Owner.m_textureBrickOverlay, 
                                           Owner.m_textureBrickMask, Owner.m_textureText};
                int offset = 0;
                for (int i = 0; i < textures.Length; i++ )
                {
                    Bitmap bitmap = Owner.m_bitmapTable[textureTypes[i]];

                    textures[i].SizeInPixels = new int2(bitmap.Width, bitmap.Height);
                    textures[i].BitmapPtr = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset;
                    textures[i].DataOffset = offset;

                    offset += FillWithChannelsFromBitmap(bitmap, Owner.Bitmaps.Host, offset);   
                }

                Owner.Bitmaps.SafeCopyToDevice();
            }
        }

        /// <summary>
        /// Renders the visible area. Not needed for simulation.
        /// </summary>
        public class RenderTask : MyTask<TetrisWorld>
        {
            
            private MyCudaKernel m_RgbaTextureKernel2DBlock;
            private MyCudaKernel m_MaskedColorKernel2DBlock;
            private int m_lastScore;
            // Streams are used for an asynchronous drawing of individual cells. 
            // They provide a 3x speedup compared to the default stream.
            CudaStream[] m_streams = new CudaStream[5]; // testing suggests values between 2 and 5 work best

            public override void Init(int nGPU)
            {
                m_RgbaTextureKernel2DBlock = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaTextureKernel2DBlock");
                m_MaskedColorKernel2DBlock = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawMaskedColorKernel2DBlock");
                m_lastScore = -1;
                for (int i = 0; i < m_streams.Length; i++)
                {
                    m_streams[i] = new CudaStream(CUStreamFlags.NonBlocking);
                }
            }

            /// <summary>
            /// Used for drawing "Score", "Level" and "Next" text into the bitmap displayed in VisualOutput.
            /// </summary>
            /// <param name="text"></param>
            /// <param name="bitmap"></param>
            /// <param name="yOffset"></param>
            private void DrawTextToBitmap(string text, Bitmap bitmap, int yOffset)
            {
                Rectangle textRect = new Rectangle(2, yOffset, 180, 22);
                Graphics g = Graphics.FromImage(bitmap);
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawString(text, new Font("Segoe UI", 12, FontStyle.Bold), Brushes.Black, textRect);
                g.Flush();
            }

            /// <summary>
            /// Draws a single cell with the specified color on the specified grid position
            /// TODO: if performance is needed, try moving this function and its caller loop to a cuda kernel,
            ///       because due to small texture sizes, the occupancy is low for these kernels (20%).
            ///       However, since asynchronous streams are used, a higher occupancy might not mean a higher speed.
            /// </summary>
            /// <param name="s">An [asynchronous] stream that the kernel will run in.</param>
            /// <param name="row"></param>
            /// <param name="column"></param>
            /// <param name="xTopLeft"></param>
            /// <param name="yTopLeft"></param>
            /// <param name="color"></param>
            private void DrawCellAtPosition(CudaStream s, int row, int column, int xTopLeft, int yTopLeft, Color color)
            {
                CudaTexture textureBrickOverlay = Owner.m_textureBrickOverlay;
                CudaTexture textureBrickMask = Owner.m_textureBrickMask;

                int xOffset = column * GFX_CELL_SQUARE_WIDTH + xTopLeft;
                int yOffset = row * GFX_CELL_SQUARE_WIDTH + yTopLeft;

                // color
                int yDiv = 10;
                Debug.Assert(textureBrickMask.SizeInPixels.y % yDiv == 0);
                m_MaskedColorKernel2DBlock.SetupExecution(new dim3(textureBrickMask.SizeInPixels.x, yDiv, 1),
                    new dim3(textureBrickMask.SizeInPixels.y / yDiv, TARGET_VALUES_PER_PIXEL, 1));
                m_MaskedColorKernel2DBlock.RunAsync(s, 
                    Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, xOffset, yOffset,
                    textureBrickMask.BitmapPtr, textureBrickMask.SizeInPixels.x, textureBrickMask.SizeInPixels.y,
                    color.R / 255.0f, color.G / 255.0f, color.B / 255.0f);

                // overlay
                Debug.Assert(textureBrickOverlay.SizeInPixels.y % yDiv == 0);
                m_RgbaTextureKernel2DBlock.SetupExecution(new dim3(textureBrickOverlay.SizeInPixels.x, yDiv, 1),
                    new dim3(textureBrickOverlay.SizeInPixels.y / yDiv, TARGET_VALUES_PER_PIXEL, 1));
                m_RgbaTextureKernel2DBlock.RunAsync(s, 
                    Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, xOffset, yOffset,
                    textureBrickOverlay.BitmapPtr, textureBrickOverlay.SizeInPixels.x, textureBrickOverlay.SizeInPixels.y);
            }

            
            /// <summary>
            /// draws tetris game board into Owner's VisualOutput memory block
            /// data is ready for rendering thanks to updateTask (score, level, brickarea, ...)
            /// performance analysis: background takes 2.7ms to draw, cells take 1.85 ms to draw (in debug)
            /// </summary>
            public override void Execute()
            {
                // erease background
                Debug.Assert(Owner.VisualHeight % 2 == 0);
                Debug.Assert(Owner.VisualWidth * 2 <= 1024); // because of blockDims:
                m_RgbaTextureKernel2DBlock.SetupExecution(new dim3(Owner.VisualWidth,2,1),
                    new dim3(Owner.VisualHeight/2, TARGET_VALUES_PER_PIXEL, 1));
                m_RgbaTextureKernel2DBlock.Run(Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, 0, 0,
                    Owner.m_textureBackground.BitmapPtr, Owner.m_textureBackground.SizeInPixels.x,
                    Owner.m_textureBackground.SizeInPixels.y);

                int iStreamIndex = 0;
                // render bricks
                for(int iRow = 2; iRow < Owner.BrickAreaRows; iRow++) // do not render rows 0 and 1
                {
                    for(int iCol = 0; iCol < Owner.BrickAreaColumns; iCol++)
                    {
                        int index = iRow * Owner.BrickAreaColumns + iCol;
                        BrickType brick = (BrickType)Math.Round(Owner.BrickAreaOutput.Host[index]);
                        if (brick != BrickType.None)
                        {
                            iStreamIndex++;
                            DrawCellAtPosition(m_streams[iStreamIndex % m_streams.Length], iRow, iCol, Owner.GFX_BRICK_AREA_TOP_LEFT.X,
                                Owner.GFX_BRICK_AREA_TOP_LEFT.Y, Owner.GetBrickColor(brick));
                        }
                    }
                }

                // render hint bricks
                for (int iRow = 0; iRow < Owner.HintAreaRows; iRow++) 
                {
                    for (int iCol = 0; iCol < Owner.HintAreaColumns; iCol++)
                    {
                        int index = iRow * Owner.HintAreaColumns + iCol;
                        BrickType brick = (BrickType)Math.Round(Owner.HintAreaOutput.Host[index]);
                        if (brick != BrickType.None)
                        {
                            iStreamIndex++;
                            DrawCellAtPosition(m_streams[iStreamIndex % m_streams.Length], iRow + 1, iCol + 1, Owner.GFX_HINT_AREA_TOP_LEFT.X,
                                Owner.GFX_HINT_AREA_TOP_LEFT.Y, Owner.GetBrickColor(brick));
                        }
                    }
                }

                // draw score and level
                // 1) update text if score changed
                int score = (int)Math.Round(Owner.ScoreOutput.Host[0]);
                if (m_lastScore != score)
                {
                    m_lastScore = score;
                    // 2) clone bitmap
                    Bitmap bitmap = Owner.m_bitmapTable[TextureType.TextArea];
                    Rectangle cloneRect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
                    PixelFormat pixelFormat = bitmap.PixelFormat;
                    bitmap = bitmap.Clone(cloneRect, pixelFormat);
                    // 3) draw text
                    DrawTextToBitmap("Score: " + score, bitmap, 0);
                    DrawTextToBitmap("Level: " + (int)Math.Round(Owner.LevelOutput.Host[0]), bitmap, 30);
                    DrawTextToBitmap("Next:", bitmap, 60);
                    int dataSize = FillWithChannelsFromBitmap(bitmap, Owner.Bitmaps.Host, Owner.m_textureText.DataOffset);
                    Owner.Bitmaps.SafeCopyToDevice(Owner.m_textureText.DataOffset, dataSize);
                }
                // 3) draw on GPU
                m_RgbaTextureKernel2DBlock.SetupExecution(new dim3(Owner.m_textureText.SizeInPixels.x, 1, 1),
                    new dim3(Owner.m_textureText.SizeInPixels.y, TARGET_VALUES_PER_PIXEL, 1));
                m_RgbaTextureKernel2DBlock.Run(Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight,
                    Owner.GFX_TEXT_AREA_TOP_LEFT.X, Owner.GFX_TEXT_AREA_TOP_LEFT.Y,
                    Owner.m_textureText.BitmapPtr, Owner.m_textureText.SizeInPixels.x, Owner.m_textureText.SizeInPixels.y);
            }
        }

        /// <summary>
        /// Update the world state based on actions, publish the new state.
        /// </summary>
        public class UpdateTask : MyTask<TetrisWorld>
        {

            // use this variable to avoid interpreting unitialized data at the first step of the simulation as input
            private bool m_firstStep;

            public override void Init(int nGPU)
            {
                m_firstStep = true;
            }

            public override void Execute()
            {
                // reset the world event output; the engine can set a new world event output, but it will only last 1 step
                Owner.WorldEventOutput.Fill(0.0f);

                ActionInputType input = DecodeAction();
 
                // The engine updates the world's memory blocks based on agent's input
                Owner.Engine.Step(input);
            }

            protected ActionInputType DecodeAction()
            {
                if(m_firstStep)
                {
                    m_firstStep = false;
                    return ActionInputType.NoAction;
                }

                if(Owner.ActionInput == null || Owner.ActionInput.Count == 0)
                {
                    return ActionInputType.NoAction;
                }

                Owner.ActionInput.SafeCopyToHost();

                // there are 6 actions possible
                if (Owner.ActionInputModality == InputModality.Number) // input is an index into the table of actions
                {
                    float fAct = Owner.ActionInput.Host[0];
                    int action = (int)Math.Round(fAct);
                    if (action > 5 || action < 0)
                        action = 0;
                    return (ActionInputType)action;
                }
                else // action is the most preferred action from the input vector of action selections
                {
                    int maxIndex = 0;
                    float max = float.NegativeInfinity;
                    for(int i = 0; i < 6; i++)
                    {
                        if(Owner.ActionInput.Host[i] > max)
                        {
                            maxIndex = i;
                            max = Owner.ActionInput.Host[i];
                        }
                    }
                    return (ActionInputType)maxIndex;
                }
            }
        }
    }
}
