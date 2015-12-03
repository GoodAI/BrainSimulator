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
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using YAXLib;

namespace GoodAI.Modules.MastermindWorld
{
    /// <author>GoodAI</author>
    /// <meta>mp</meta>
    /// <status>Working</status>
    /// <summary> World for the Mastermind game.</summary>
    /// <description>
    /// <p>
    /// World simulating the <a href="https://en.wikipedia.org/wiki/Mastermind_(board_game)">Mastermind</a> game (aka bulls and cows). <br />
    /// The game allows the agent to make at most m (typically, m = 12) guesses to find out 
    /// a hidden vector V of k colours. Typically, k is 4 or 5. A guess is a vector G of k colours. The same colours can repeat in one vector.<br />
    /// The agent wins if he correctly guesses the vector V (that is, G = V) before he runs out of guesses.<br />
    /// The world tells the agent how well it made the last guess by outputing the number of bulls B (correctly guessed colors on correct positions)
    /// and the number of cows W (correctly guessed colors, but on wrong positions) for the last guess.
    /// After the agent wins or makes m (all available) guesses, the game resets. A new hidden vector V may be set, depending on user input.<br />
    /// The hidden vector V can be specified as user input. If user input is empty, vector V is random. <br />
    /// </p>
    /// 
    /// <p>
    /// The agent can, at any timestep:
    /// <ul>
    ///     <li>Output a guess G.</li>
    ///     <li>Confirm the guess G (tell the world that G currently on the ouptut is the guess he wants to make).</li>
    /// </ul>
    /// </p>
    /// 
    /// The world is composed of:
    /// <ul>
    ///     <li>The agent.</li>
    ///     <li>The set of available colours <b>C</b>.</li>
    ///     <li>The hidden vector <b>V</b> of k colours; <b>V &#8712; C<sup>k</sup></b> .</li>
    ///     <li>The history of <b>n</b> confirmed agent guesses 
    ///         <b>G<sup>n</sup></b>, <b>G<sub>i</sub> &#8712; C<sup>k</sup></b> &#8704; 0 &#8804; i &#60; n.</li>
    ///     <li>The history of <b>n</b> world's evaluations of confirmed agent guesses 
    ///         <b>(B,W)<sup>n</sup></b>, <b>(B,W)<sub>i</sub> &#8712; N x N</b> &#8704; 0 &#8804; i &#60; n (number of bulls and cows for each guess).</li>
    ///     <li>A reward signal that is 1 when the agent correctly guesses vector V.</li>
    ///     <!--<li>A reset button (todo).</li>-->
    /// </ul>
    /// 
    /// The world resets when:
    /// <ul>
    ///     <li>The agent makes m guesses.</li>
    ///     <li>The agent correctly guesses hidden vector V.</li>
    /// </ul>
    /// 
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li><b>Number of available colours:</b> <b>|C|</b>. In other words, number of different colours that can be elements 
    ///         of vectors V and G. By default, |C|=6.</li>
    ///     <li><b>Length of hidden vector:</b> number <b>k</b> of elements in vector V. By default, k=4.</li>
    ///     <li><b>Number of guesses:</b> number <b>m</b> of guesses the agent can make before the game resets. By default, m=12.</li>
    ///     <li><b>Repeating colors:</b> when true, allows the same color to appear multiple times in the hidden vector.</li>
    ///     <li><b>Hidden vector:</b> the vector that the agent should guess. If empty, a random vector is generated for each game.</li>
    ///     <li><b>Repeatable hidden vector:</b> when true, the random generator is re-set every time the simulation is stopped.
    ///         This means that the sequence of randomly generated hidden vectors is repeatable.</li>
    /// </ul>
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li><b>GuessInput:</b> vector G of k colors that the agent submits as a guess.</li>
    ///     <li><b>ActionInput:</b> a number (0 or 1) that indicates whether the current guess G is being confirmed by the agent. 
    ///         This input allows the agent to think multiple computation steps about the correct guess, before providing it to the world.
    ///         0 means a confirmation of the guess, 1 means the world will ignore the guess.</li>
    /// </ul>
    /// 
    /// <h3>Outputs</h3>
    /// <ul>
    ///     <li><b>HiddenVectorOutput:</b> the hidden vector V of k colours that the agent should find out.</li>
    ///     <li><b>GuessCountOutput:</b> the number of guesses the agent has confirmed since the last world reset.</li>
    ///     <li><b>GuessesOutput:</b> an array of confirmed guesses that the agent has made since the last world reset. 
    ///         The array has a capacity of <b>m</b> guesses.</li>
    ///     <li><b>GuessEvaluationsOutput:</b> an array of bulls and cows, which the world computed for each agent's confirmed guess. 
    ///         The array has a capacity of <b>m</b> pairs (bull,cow).</li>
    ///     <li><b>WorldEventOutput:</b> a number that indicates different world events that occur at each time step: 
    ///         0 = no event, 1 = agent won + reset, -1 = agent lost + reset</li>
    ///     <li><b>VisualOutput:</b> a bitmap that represents the current world state.</li>
    /// </ul>
    /// 
    /// </description>
    public class MyMastermindWorld : MyWorld
    {

        // implementation note: the world is tightly coupled with its tasks and with the engine.

        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> GuessInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> ActionInput
        {
            get { return GetInput(1); }
        }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> HiddenVectorOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> GuessCountOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> GuessHistoryOutput
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> GuessEvaluationHistoryOutput
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> WorldEventOutput
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        [MyOutputBlock(5)]
        public MyMemoryBlock<float> VisualOutput
        {
            get { return GetOutput(5); }
            set { SetOutput(5, value); }
        }

        public MyMemoryBlock<float> Bitmaps { get; private set; }

        #endregion

        private Dictionary<MyMastermindWorld.TextureType, Bitmap> m_bitmapTable = new Dictionary<MyMastermindWorld.TextureType, Bitmap>();
        protected MastermindWorldEngine m_engine;
        protected MyWorldEngineParams m_engineParams = new MyWorldEngineParams();

        #region Parameters

        [MyBrowsable, Category("Complexity"), Description("Number of elements in the hidden vector.")]
        [YAXSerializableField(DefaultValue = 4)]
        public int HiddenVectorLength {
            get { return m_engineParams.HiddenVectorLength; }
            set { m_engineParams.HiddenVectorLength = value; }
        } // limited by validation

        [/*MyBrowsable,*/ Category("Complexity constants")]
        [YAXSerializableField(DefaultValue = 1)]
        public int HiddenVectorLengthMin { get; protected set; }

        [/*MyBrowsable,*/ Category("Complexity constants")]
        [YAXSerializableField(DefaultValue = 20)]
        public int HiddenVectorLengthMax { get; protected set; }

        [MyBrowsable, Category("Complexity"), Description("Number of colors to choose from for each element of the hidden vector.")]
        [YAXSerializableField(DefaultValue = 6)]
        public int NumberOfColors {
            get { return m_engineParams.NumberOfColors; }
            set { m_engineParams.NumberOfColors = value; }
        } // limited by validation

        [/*MyBrowsable,*/ Category("Complexity constants")]
        [YAXSerializableField(DefaultValue = 2)]
        public int NumberOfColorsMin { get; protected set; }

        [/*MyBrowsable,*/ Category("Complexity constants")]
        [YAXSerializableField(DefaultValue = 100)]
        public int NumberOfColorsMax { get; protected set; }

        [MyBrowsable, Category("Complexity"), Description("Number of guesses per game.")]
        [YAXSerializableField(DefaultValue = 12)]
        public int NumberOfGuesses {
            get { return m_engineParams.NumberOfGuesses; }
            set { 
                m_engineParams.NumberOfGuesses = value;
                m_engineParams.NumberOfRenderedGuesses = Math.Max(NumberOfGuessesMin, Math.Min(value, 20)); // will not render more than 20 guesses
            }
        } // limited by validation

        [/*MyBrowsable,*/ Category("Complexity constants")]
        [YAXSerializableField(DefaultValue = 1)]
        public int NumberOfGuessesMin { get; protected set; }

        [/*MyBrowsable,*/ Category("Complexity constants")]
        [YAXSerializableField(DefaultValue = 100000)]
        public int NumberOfGuessesMax { get; protected set; }

        [Category("Constants")]
        [YAXSerializableField(DefaultValue = 100)]
        public int VisualWidth { get; protected set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 100)]
        public int VisualHeight { get; protected set; }

        [MyBrowsable, Category("Start configuration"), Description("If left empty, a random hidden vector is generated for each game.")]
        [YAXSerializableField(DefaultValue = "")]
        public string HiddenVectorUser {
            get { return m_engineParams.HiddenVectorUser; }
            set { m_engineParams.HiddenVectorUser = value; }
        }

        [MyBrowsable, Category("Start configuration")]
        [YAXSerializableField(DefaultValue = true)]
        public bool CanRepeatColors {
            get { return m_engineParams.CanRepeatColors; }
            set { m_engineParams.CanRepeatColors = value; }
        } // checked at validation if the user input of hidden vector matches this setting

        [MyBrowsable, Category("Start configuration"), Description("If true, the sequence of random hidden vectors repeats each time the simulation is restarted.")]
        [YAXSerializableField(DefaultValue = false)]
        public bool RepeatableHiddenVector
        {
            get { return m_engineParams.RepeatableHiddenVector; }
            set { m_engineParams.RepeatableHiddenVector = value; }
        }

        #endregion

        #region Graphics parameters

        private string textureSet = @"res\mastermindworld\";
        private const int GFX_MAIN_SPACER = 4*5;
        private const int GFX_EVALUATION_SPACER = 4*1;
        private const int GFX_GUESS_SPACER = 4*2;
        private const int SOURCE_VALUES_PER_PIXEL = 4; // RGBA: 4 floats per pixel
        private const int TARGET_VALUES_PER_PIXEL = 3; // RGB: 3 floats per pixel

        #endregion

        /// <summary>
        /// Number of floats needed to store a single evaluation of a guess. 1 for bulls + 1 for cows
        /// </summary>
        public const int EVALUATION_ITEM_LENGTH = 2;

        /// <summary>
        /// Actions that the agent can submit to the world's ActionInput
        /// </summary>
        public enum ActionInputKind
        {
            ConfirmGuess = 0,
            NoAction = 1
        }

        /// <summary>
        /// Three relations that an element of a guess can be in w.r.t. the hidden vector
        /// </summary>
        public enum EvaluationKind
        {
            Bull = 0,
            Cow = 1,
            Miss = 2
        }

        public enum TextureType
        {
            CircleRim = 0,
            CircleMask = 1,
            Miss = 2,
            Bull = 3,
            Cow = 4
        }

        public MyMastermindWorld()
        {
            LoadBitmap(TextureType.CircleRim, textureSet + "circle_rim.png");
            LoadBitmap(TextureType.CircleMask, textureSet + "circle_mask.png");
            LoadBitmap(TextureType.Miss, textureSet + "miss.png");
            LoadBitmap(TextureType.Bull, textureSet + "bull.png");
            LoadBitmap(TextureType.Cow, textureSet + "cow.png");
        }
        
        /// <summary>
        /// Called before the start of the simulation. The sizes of the memory blocks depend on the variables from
        /// the code region "Parameters".
        /// </summary>
        public override void UpdateMemoryBlocks()
        {
            Bitmaps.Count = 0;
            Bitmaps.Count += GetBitmapSize(TextureType.CircleRim);
            Bitmaps.Count += GetBitmapSize(TextureType.CircleMask);
            Bitmaps.Count += GetBitmapSize(TextureType.Miss);
            Bitmaps.Count += GetBitmapSize(TextureType.Bull);
            Bitmaps.Count += GetBitmapSize(TextureType.Cow);

            HiddenVectorOutput.Count = HiddenVectorLength;
            HiddenVectorOutput.ColumnHint = HiddenVectorLength;

            GuessCountOutput.Count = 1;
            
            GuessHistoryOutput.Count = (NumberOfGuesses * HiddenVectorLength);
            GuessHistoryOutput.ColumnHint = HiddenVectorLength;

            GuessEvaluationHistoryOutput.Count = (NumberOfGuesses * EVALUATION_ITEM_LENGTH);
            GuessEvaluationHistoryOutput.ColumnHint = EVALUATION_ITEM_LENGTH;

            WorldEventOutput.Count = 1;

            int evaluationSquareWidth, guessCircleWidth, evaluationWidth,
                evaluationsPerRow, guessWidth, rowWidth, rowHeight;
            GetGraphicsParameters(out evaluationSquareWidth, out guessCircleWidth, out evaluationWidth,
                                  out evaluationsPerRow, out guessWidth, out rowWidth, out rowHeight);
            VisualWidth = rowWidth;
            VisualHeight = rowHeight * m_engineParams.NumberOfRenderedGuesses;
            VisualOutput.Count = VisualWidth * VisualHeight * TARGET_VALUES_PER_PIXEL;
            VisualOutput.ColumnHint = VisualWidth;
        }

        /// <summary>
        /// Returns pixel dimensions and distances of important elements on the game board.
        /// </summary>
        /// <param name="evaluationSquareWidth">pixel width (and height) of the textures that are used for displaying a bull/cow/miss</param>
        /// <param name="guessCircleWidth">pixel width (and height) of the texture that is used for displaying a circle of a guess</param>
        /// <param name="evaluationWidth">total pixel width of the group of bulls/cows/misses</param>
        /// <param name="evaluationsPerRow">maximum number of bulls/cows/misses that appear on a single line</param>
        /// <param name="guessWidth">total pixel width of the sequence of circles that forms a guess</param>
        /// <param name="rowWidth">total pixel width of (a single row of) a game board</param>
        /// <param name="rowHeight">total pixel height of a single row of a game board</param>
        public void GetGraphicsParameters(out int evaluationSquareWidth, out int guessCircleWidth, out int evaluationWidth,
            out int evaluationsPerRow, out int guessWidth, out int rowWidth, out int rowHeight)
        {
            evaluationSquareWidth = m_bitmapTable[TextureType.Bull].Width;
            guessCircleWidth = m_bitmapTable[TextureType.CircleRim].Width;
            evaluationsPerRow = (int)Math.Ceiling(HiddenVectorLength / 2.0); // evaluations come in two rows

            evaluationWidth = evaluationsPerRow * evaluationSquareWidth + (evaluationsPerRow - 1) * GFX_EVALUATION_SPACER;
            guessWidth = HiddenVectorLength * guessCircleWidth + (HiddenVectorLength - 1) * GFX_GUESS_SPACER;
            rowWidth = GFX_MAIN_SPACER + evaluationWidth + GFX_MAIN_SPACER + guessWidth + GFX_MAIN_SPACER;
            rowHeight = m_bitmapTable[TextureType.CircleRim].Height + GFX_MAIN_SPACER;
        }


        protected Dictionary<int, Color> m_colorCodes = new Dictionary<int,Color>();
        protected Random randomColor = new Random(0); // same random colors each run
        /// <summary>
        /// called from the init task, to create an association between color codes and colors themselves
        /// </summary>
        public void GenerateGuessColors()
        {
            for(int i = 0; i < NumberOfColors; i++)
            {
                if (m_colorCodes.ContainsKey(i)) // keeps colors accross runs
                    continue;
                Color c;
                switch(i) // first 6 colors are preset, the rest is random
                {
                    case 0:
                        c = Color.FromArgb(200,0,0); // red
                        break;
                    case 1:
                        c = Color.FromArgb(255, 153, 63); // orange
                        break;
                    case 2:
                        c = Color.FromArgb(78, 193, 40); // green
                        break;
                    case 3:
                        c = Color.FromArgb(100, 100, 255); // blue
                        break;
                    case 4:
                        c = Color.FromArgb(255, 255, 43); // yellow
                        break;
                    case 5:
                        c = Color.FromArgb(255, 58, 176); // violet
                        break;
                    default:
                        c = Color.FromArgb(randomColor.Next(256), randomColor.Next(256), randomColor.Next(256));
                        break;
                }
                m_colorCodes[i] = c;
            }
        }

        /// <summary>
        /// Returns the color associated with color code i
        /// </summary>
        /// <param name="i">the number whose color we want</param>
        /// <returns></returns>
        public Color GetGuessColor(int i)
        {
            if (i < m_colorCodes.Count)
                return m_colorCodes[i];

            return Color.FromArgb(0, 0, 0); // unknown color
        }

        /// <summary>
        /// Loads a bitmap from a file and stores it in a dictionary. Checks for ARGB color format (e.g. 32bit png). 
        /// Implementation similar to MyGridWorld.
        /// </summary>
        /// <param name="textureType"></param>
        /// <param name="path"></param>
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
        /// </summary>
        /// <param name="textureType"></param>
        /// <returns></returns>
        protected int GetBitmapSize(MyMastermindWorld.TextureType textureType)
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
        /// Implementation similar to MyGridWorld.
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
            if (value > valueMax)
            {
                validator.AddError(this, string.Format("{0} cannot be more than {1}.", valueName, valueMax));
            }
            if (value < valueMin)
            {
                validator.AddError(this, string.Format("{0} cannot be less than {1}.", valueName, valueMin));
            }
        }

        private void ValidateHiddenVectorUser(MyValidator validator)
        {
            if (m_engineParams.HiddenVectorUserParsed != null)
            {
                // must have a correct format and must be empty or contain HiddenVectorLength numerical entries; 
                // the number of unique entries must be less or equal to the number of colours and it must match the CanRepeatColors property
                if (m_engineParams.HiddenVectorUserParsed.Length != HiddenVectorLength)
                {
                    validator.AddError(this, "Number of elements specified in HiddenVectorUser must match HiddenVectorLength.");
                }
                if (!CanRepeatColors)
                {
                    HashSet<int> userPicked = new HashSet<int>(m_engineParams.HiddenVectorUserParsed);
                    if (userPicked.Count != m_engineParams.HiddenVectorUserParsed.Length)
                    {
                        validator.AddError(this, "CanRepeatColors is false, but HiddenVectorUser contains repeating colors.");
                    }
                }
                // check color codes in user input
                for (int i = 0; i < m_engineParams.HiddenVectorUserParsed.Length; i++)
                {
                    if (m_engineParams.HiddenVectorUserParsed[i] >= NumberOfColors)
                    {
                        validator.AddError(this, string.Format("HiddenVectorUser must not contain color codes greater or equal to NumberOfColors ({0}).", 
                            NumberOfColors));
                        break;
                    }
                    if (m_engineParams.HiddenVectorUserParsed[i] < 0)
                    {
                        validator.AddError(this, string.Format("HiddenVectorUser must not contain color codes less than 0."));
                        break;
                    }
                }
            }

            if (GuessInput != null && HiddenVectorLength != GuessInput.Count)
            {
                validator.AddError(this, "Number of elements in the GuessInput memory block must match HiddenVectorLength.");
            }

            if (!CanRepeatColors && NumberOfColors < HiddenVectorLength)
            {
                validator.AddError(this, "Repeating of colors in the hidden vector is forbidden: NumberOfColors cannot be lower than HiddenVectorLength.");
            }
        }

        public override void Validate(MyValidator validator)
        {
            ValidateRange(validator, HiddenVectorLength, HiddenVectorLengthMin, HiddenVectorLengthMax, "HiddenVectorLength");
            ValidateRange(validator, NumberOfColors, NumberOfColorsMin, NumberOfColorsMax, "NumberOfColors");
            ValidateRange(validator, NumberOfGuesses, NumberOfGuessesMin, NumberOfGuessesMax, "NumberOfGuesses");
            ValidateHiddenVectorUser(validator);
        }

        public MyInitTask InitGameTask { get; protected set; }
        public MyUpdateTask UpdateTask { get; protected set; }
        public MyRenderTask RenderGameTask { get; protected set; }

        public class MyCudaTexture // based on MyGridWorld.MyGraphicsPrototype
        {
            public int2 SizeInPixels;
            public CUdeviceptr BitmapPtr;
        }

        protected MyCudaTexture m_textureMiss = new MyCudaTexture(), 
            m_textureCow = new MyCudaTexture(), 
            m_textureBull = new MyCudaTexture(), 
            m_textureCircleRim = new MyCudaTexture(), 
            m_textureCircleMask = new MyCudaTexture();

        /// <summary>
        /// Initialize the world (load graphics, create engine).
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MyMastermindWorld>
        {
            public override void Init(int nGPU)
            {
                Owner.m_engine = new MastermindWorldEngine(Owner, Owner.m_engineParams);
            }

            public override void Execute()
            {
                Owner.m_engine.Reset();

                Owner.GenerateGuessColors();

                // load bitmaps to Bitmaps memory block; remember their offsets in Owner's texture variables
                CudaDeviceVariable<float> devBitmaps = Owner.Bitmaps.GetDevice(Owner);
                TextureType[] textureTypes = { TextureType.CircleRim, TextureType.CircleMask, TextureType.Miss, 
                                                 TextureType.Cow, TextureType.Bull };
                MyCudaTexture[] textures = { Owner.m_textureCircleRim, Owner.m_textureCircleMask, Owner.m_textureMiss, 
                                               Owner.m_textureCow,  Owner.m_textureBull};
                int offset = 0;
                for (int i = 0; i < textures.Length; i++ )
                {
                    Bitmap bitmap = Owner.m_bitmapTable[textureTypes[i]];

                    textures[i].SizeInPixels = new int2(bitmap.Width, bitmap.Height);
                    textures[i].BitmapPtr = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset;

                    offset += FillWithChannelsFromBitmap(bitmap, Owner.Bitmaps.Host, offset);   
                }

                Owner.Bitmaps.SafeCopyToDevice();
            }
        }

        /// <summary>
        /// Renders the visible area. Not needed for simulation.<br />
        /// When using an observer to visualize the output, choose RGB as Coloring Method.
        /// 
        /// </summary>
        public class MyRenderTask : MyTask<MyMastermindWorld>
        {
            
            private MyCudaKernel m_RgbaTextureKernel;
            private MyCudaKernel m_MaskedColorKernel;
            private MyCudaKernel m_RgbBackgroundKernel;
            CudaStream[] m_streams = new CudaStream[5];

            public override void Init(int nGPU)
            {
                m_RgbBackgroundKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbBackgroundKernel");

                m_RgbaTextureKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaTextureKernel2DBlock");

                m_MaskedColorKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawMaskedColorKernel2DBlock");

                for(int i = 0; i < m_streams.Length; i++)
                {
                    m_streams[i] = new CudaStream();
                }
            }

            /// <summary>
            /// Draws a single element of a guess at the provided position
            /// </summary>
            /// <param name="s">[async] stream that the kernels will execute in</param>
            /// <param name="guess">element of a guess to draw</param>
            /// <param name="position">the positiion (0,1,2,...) of the element inside the guess vector</param>
            /// <param name="xGroupOffset">X pixel offset of the first element of the guess vector</param>
            /// <param name="yGroupOffset">Y pixel offset of the first element of the guess vector</param>
            private void DrawGuessAtPosition(CudaStream s, float guess, int position, int xGroupOffset, int yGroupOffset)
            {
                MyCudaTexture textureCircleRim = Owner.m_textureCircleRim;
                MyCudaTexture textureCircleMask = Owner.m_textureCircleMask;

                int xOffset = xGroupOffset + position * (GFX_GUESS_SPACER + textureCircleRim.SizeInPixels.x);
                int yOffset = yGroupOffset;
                Color guessColor = Owner.GetGuessColor((int)Math.Round(guess));

                // color
                int yDiv = 10;
                Debug.Assert(textureCircleMask.SizeInPixels.y % yDiv == 0);
                m_MaskedColorKernel.SetupExecution(new dim3(textureCircleMask.SizeInPixels.x, yDiv, 1),
                                                   new dim3(textureCircleMask.SizeInPixels.y / yDiv, TARGET_VALUES_PER_PIXEL));
                m_MaskedColorKernel.RunAsync(s, Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, xOffset, yOffset,
                    textureCircleMask.BitmapPtr, textureCircleMask.SizeInPixels.x, textureCircleMask.SizeInPixels.y,
                    guessColor.R / 255.0f, guessColor.G / 255.0f, guessColor.B / 255.0f);

                // circle rim 
                Debug.Assert(textureCircleRim.SizeInPixels.y % yDiv == 0);
                m_RgbaTextureKernel.SetupExecution(new dim3(textureCircleRim.SizeInPixels.x, yDiv, 1),
                                                   new dim3(textureCircleRim.SizeInPixels.y / yDiv, TARGET_VALUES_PER_PIXEL));
                m_RgbaTextureKernel.RunAsync(s, Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, xOffset, yOffset,
                    textureCircleRim.BitmapPtr, textureCircleRim.SizeInPixels.x, textureCircleRim.SizeInPixels.y);
            }

            /// <summary>
            /// Works just like DrawGuessAtPosition, only for evaluations.
            /// </summary>
            private void DrawEvaluationAtPosition(CudaStream s, MyMastermindWorld.EvaluationKind evaluation, int position, 
                int xGroupOffset, int yGroupOffset, int evaluationsPerRow)
            {
                MyCudaTexture texture;
                switch(evaluation)
                {
                    case MyMastermindWorld.EvaluationKind.Bull:
                        texture = Owner.m_textureBull;
                        break;
                    case MyMastermindWorld.EvaluationKind.Cow:
                        texture = Owner.m_textureCow;
                        break;
                    case MyMastermindWorld.EvaluationKind.Miss:
                    default:
                        texture = Owner.m_textureMiss;
                        break;
                }

                int xOffset = xGroupOffset + (position % evaluationsPerRow) * (GFX_EVALUATION_SPACER + texture.SizeInPixels.x);
                int yOffset = yGroupOffset + (position / evaluationsPerRow) * (GFX_EVALUATION_SPACER + texture.SizeInPixels.y);

                int yDiv = 10;
                Debug.Assert(texture.SizeInPixels.y % yDiv == 0);
                m_RgbaTextureKernel.SetupExecution(new dim3(texture.SizeInPixels.x, yDiv, 1),
                                                   new dim3(texture.SizeInPixels.y / yDiv, TARGET_VALUES_PER_PIXEL));
                m_RgbaTextureKernel.RunAsync(s, Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, xOffset, yOffset,
                    texture.BitmapPtr, texture.SizeInPixels.x, texture.SizeInPixels.y);
            }

            
            /// <summary>
            /// draws mastermind game board into Owner's VisualOutput memory block
            /// </summary>
            public override void Execute()
            {
                // erease background
                int blockDimX = Owner.VisualWidth;
                int gridDimX = Owner.VisualHeight;
                if (blockDimX > 1024)
                {
                    gridDimX *= (int)(Math.Ceiling(blockDimX / 1024.0));
                    blockDimX = 1024;
                }
                m_RgbBackgroundKernel.SetupExecution(new dim3(blockDimX, 1, 1), new dim3(gridDimX, TARGET_VALUES_PER_PIXEL, 1));
                m_RgbBackgroundKernel.Run(Owner.VisualOutput, Owner.VisualWidth, Owner.VisualHeight, 0.9f, 0.9f, 0.9f);

                // prepare data for rendering
                Owner.GuessHistoryOutput.SafeCopyToHost();
                Owner.GuessEvaluationHistoryOutput.SafeCopyToHost();
                Owner.GuessCountOutput.SafeCopyToHost();

                int evaluationSquareWidth, guessCircleWidth, evaluationWidth,
                    evaluationsPerRow, guessWidth, rowWidth, rowHeight;
                Owner.GetGraphicsParameters(out evaluationSquareWidth, out guessCircleWidth, out evaluationWidth,
                    out evaluationsPerRow, out guessWidth, out rowWidth, out rowHeight);
                int xGuessGroupOffset = GFX_MAIN_SPACER + evaluationWidth + GFX_MAIN_SPACER;
                int xEvaluationGroupOffset = GFX_MAIN_SPACER;
                int totalGuesses = (int)Math.Round(Owner.GuessCountOutput.Host[0]);

                int iStreamIndex = 0;
                // render at most NumberOfRenderedGuesses. The latest guesses have priority.
                for (int iGuess = 0; iGuess < Math.Min(totalGuesses, Owner.m_engineParams.NumberOfRenderedGuesses); iGuess++)
                {
                    int iGuessIndex = iGuess;
                    if(totalGuesses > Owner.m_engineParams.NumberOfRenderedGuesses)
                    {
                        iGuessIndex += totalGuesses - Owner.m_engineParams.NumberOfRenderedGuesses;
                    }

                    int yGuessGroupOffset = GFX_MAIN_SPACER + iGuess * (guessCircleWidth + GFX_MAIN_SPACER);
                    for (int iPosition = 0; iPosition < Owner.HiddenVectorLength; iPosition++)
                    {
                        iStreamIndex++;
                        DrawGuessAtPosition(m_streams[iStreamIndex % m_streams.Length], 
                            Owner.GuessHistoryOutput.Host[iGuessIndex*Owner.HiddenVectorLength + iPosition], 
                            iPosition, xGuessGroupOffset, yGuessGroupOffset);
                    }

                    int yEvaluationGroupOffset = yGuessGroupOffset + GFX_EVALUATION_SPACER;
                    int bulls = (int)Math.Round(Owner.GuessEvaluationHistoryOutput.Host[iGuessIndex * MyMastermindWorld.EVALUATION_ITEM_LENGTH + 0]);
                    int cows = (int)Math.Round(Owner.GuessEvaluationHistoryOutput.Host[iGuessIndex * MyMastermindWorld.EVALUATION_ITEM_LENGTH + 1]);
                    for (int iPosition = 0; iPosition < Owner.HiddenVectorLength; iPosition++)
                    {
                        MyMastermindWorld.EvaluationKind ev = EvaluationKind.Miss;
                        if(bulls > 0)
                        {
                            ev = EvaluationKind.Bull;
                            bulls--;
                        }
                        else if (cows > 0)
                        {
                            ev = EvaluationKind.Cow;
                            cows --;
                        }
                        iStreamIndex++;
                        DrawEvaluationAtPosition(m_streams[iStreamIndex % m_streams.Length], 
                            ev, iPosition, xEvaluationGroupOffset, yEvaluationGroupOffset, 
                            evaluationsPerRow);
                    }
                }
            }
        }

        /// <summary>
        /// Update the world state based on actions, publish the new state.
        /// </summary>
        public class MyUpdateTask : MyTask<MyMastermindWorld>
        {

            // use this variable to avoid interpreting unitialized data at the first step of the simulation as a guess
            private bool m_firstStep;

            public override void Init(int nGPU)
            {
                m_firstStep = true;
            }

            public override void Execute()
            {
                // reset the world event output; the engine can set a new world event output, but it will only last 1 step
                Owner.WorldEventOutput.Fill(0.0f);

                ActionInputKind input = DecodeAction();

                // Update the state of the world if the agent has confirmed the new input. 
                // The engine updates the world's memory blocks based on agent's input
                if(input == ActionInputKind.ConfirmGuess)
                {
                    Owner.m_engine.Step();
                }
            }

            // get action with the max (utility) value
            protected ActionInputKind DecodeAction()
            {
                if(m_firstStep)
                {
                    m_firstStep = false;
                    return ActionInputKind.NoAction;
                }

                if(Owner.ActionInput == null || Owner.ActionInput.Count == 0)
                {
                    return ActionInputKind.ConfirmGuess;
                }

                Owner.ActionInput.SafeCopyToHost();
                float fAct = Owner.ActionInput.Host[0];
                if ((int)Math.Round(fAct) == 0)
                {
                    return ActionInputKind.ConfirmGuess;
                }
                return ActionInputKind.NoAction;
            }
        }
    }
}
