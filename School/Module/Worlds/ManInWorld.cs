using GoodAI.Core;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using PixelFormat = System.Drawing.Imaging.PixelFormat;

namespace GoodAI.Modules.School.Worlds
{
    /// <author>GoodAI</author>
    /// <meta>Mp,Mv,Os</meta>
    /// <status>WIP</status>
    /// <summary> Implementation of a configurable 2D world </summary>
    /// <description>
    /// Implementation of a configurable 2D world
    /// </description>
    public abstract class ManInWorld : MyWorld, IWorldAdapter
    {
        public int DegreesOfFreedom { get; set; }

        [MyOutputBlock(0), DynamicBlock]
        public MyMemoryBlock<float> VisualFOW
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> VisualPOW
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2), DynamicBlock]
        public MyMemoryBlock<float> Objects
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> Reward
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Controls
        {
            get { return GetInput(0); }
        }

        protected virtual string TEXTURE_DIR { get { return @"res\FILL_IN_INHERITED_CLASS"; } }
        protected string TEXTURE_DIR_COMMON = @"res\SchoolWorldCommon\";

        protected int m_FOW_WIDTH = 1024;
        protected int m_FOW_HEIGHT = 1024;

        public virtual int FOW_WIDTH { get { return m_FOW_WIDTH; } set { m_FOW_WIDTH = value; } }
        public virtual int FOW_HEIGHT { get { return m_FOW_HEIGHT; } set { m_FOW_HEIGHT = value; } }

        public virtual int POW_WIDTH { get { return 256; } }
        public virtual int POW_HEIGHT { get { return 256; } }

        public virtual Color BackgroundColor { get; set; }

        public const float DUMMY_PIXEL = float.NaN;
        private const int PIXEL_SIZE = 4; // RGBA: 4 floats per pixel

        public MovableGameObject Agent { get; protected set; }
        // object which should be implemented with same actions and same behavior as agent have
        public AbstractTeacherInWorld Teacher { get; set; }

        public float ContinuousAction { get; set; }

        public float Time = 1f;

        private bool m_IsWorldFrozen = false;
        public bool IsWorldFrozen { get { return m_IsWorldFrozen; } } // nothing moves in a frozen world

        // noise:
        // R/G/B intensity value range is 0-256
        public float ImageNoiseStandardDeviation = 20.0f; // the noise follows a normal distribution (maybe can be simpler?)
        public float ImageNoiseMean = 0; // the average value that is added to each pixel in the image
        public bool IsImageNoise { get; set; }

        public List<GameObject> gameObjects;

        public StandardConflictResolver ConflictResolver = new StandardConflictResolver();

        public static Size DEFAULT_GRID_SIZE = new Size(32, 32);

        // tuple contains the bitmap itself and its offset in Bitmaps memory block
        private Dictionary<string, Tuple<Bitmap, int>> m_bitmapTable;
        private int m_totalOffset;

        [DynamicBlock]
        public MyMemoryBlock<float> Bitmaps { get; protected set; }
        private string m_errorMessage;

        public MyMemoryBlock<float> AgentVisualTemp { get; protected set; } // used e.g. for holding random numbers during noise generation

        public ManInWorld()
        {
            BackgroundColor = Color.FromArgb(77, 174, 255);
            gameObjects = new List<GameObject>();
            m_bitmapTable = new Dictionary<string, Tuple<Bitmap, int>>();
            ClearWorld();
        }

        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }

        public void InitAdapterMemory(SchoolWorld schoolWorld)
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 128;
        }

        public override MyMemoryBlock<float> GetInput(int index)
        {
            if (ControlsAdapterTemp != null)  //HACK which checks if World is standalone or in SchoolWorld. TODO fix it somehow.
                return ControlsAdapterTemp as MyMemoryBlock<float>;
            else
                return base.GetInput<float>(index);
        }

        public override MyMemoryBlock<T> GetInput<T>(int index)
        {
            if (ControlsAdapterTemp != null) //HACK which checks if World is standalone or in SchoolWorld. TODO fix it somehow.
                return ControlsAdapterTemp as MyMemoryBlock<T>;
            else
                return base.GetInput<T>(index);
        }

        public override MyAbstractMemoryBlock GetAbstractInput(int index)
        {
            if (ControlsAdapterTemp != null) //HACK which checks if World is standalone or in SchoolWorld. TODO fix it somehow.
                return ControlsAdapterTemp;
            else
                return base.GetAbstractInput(index);
        }

        public virtual void InitWorldInputs(int nGPU, SchoolWorld schoolWorld)
        {

        }

        public virtual void MapWorldInputs(SchoolWorld schoolWorld)
        {
            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            ControlsAdapterTemp.CopyFromMemoryBlock(schoolWorld.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, schoolWorld.ActionInput.Count));
        }

        public virtual void InitWorldOutputs(int nGPU, SchoolWorld schoolWorld)
        {

        }

        public virtual void MapWorldOutputs(SchoolWorld schoolWorld)
        {
            // Copy data from world to wrapper
            VisualPOW.CopyToMemoryBlock(schoolWorld.Visual, 0, 0, Math.Min(VisualPOW.Count, schoolWorld.VisualSize));
            if (Objects.Count > 0)
                Objects.CopyToMemoryBlock(schoolWorld.Data, 0, 0, Math.Min(Objects.Count, schoolWorld.DataSize));
            //schoolWorld.Visual.Dims = VisualPOW.Dims;
            schoolWorld.DataLength.Fill(Math.Min(Objects.Count, schoolWorld.DataSize));
            Reward.CopyToMemoryBlock(schoolWorld.Reward, 0, 0, 1);
        }

        public virtual void ClearWorld()
        {
            Agent = null;
            gameObjects.Clear();
            Objects.Count = 0;
            IsImageNoise = false;
            m_IsWorldFrozen = false;
            DegreesOfFreedom = 2;
        }

        public void SetHint(TSHintAttribute attr, float value)
        {
            if (attr == TSHintAttributes.IMAGE_NOISE)
            {
                IsImageNoise = value > 0;
            }
            else if (attr == TSHintAttributes.DEGREES_OF_FREEDOM)
            {
                DegreesOfFreedom = (int)value;
            }
        }

        public override void UpdateMemoryBlocks()
        {
            VisualFOW.Count = FOW_WIDTH * FOW_HEIGHT;
            VisualFOW.ColumnHint = FOW_WIDTH;

            VisualPOW.Count = POW_HEIGHT * POW_WIDTH;
            VisualPOW.ColumnHint = POW_WIDTH;

            AgentVisualTemp.Count = VisualPOW.Count * 3;

            Bitmaps.Count = 0;

            gameObjects.Clear();
            m_bitmapTable.Clear();

            m_totalOffset = 0;

            Objects.Count = 0;

            Reward.Count = 1;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            validator.AssertError(POW_HEIGHT <= FOW_HEIGHT, this, "POW_HEIGHT cannot be higher than FOW_HEIGHT, corresponding sizes are: " + POW_HEIGHT + ", " + FOW_HEIGHT);
            validator.AssertError(POW_WIDTH <= FOW_WIDTH, this, "POW_WIDTH cannot be higher than FOW_WIDTH, corresponding sizes are: " + POW_WIDTH + ", " + FOW_WIDTH);
        }

        protected Point GetPowCenter()
        {
            GameObject agent = Agent;
            if (agent == null)
            {
                return new Point(FOW_WIDTH / 2, FOW_HEIGHT / 2);
            }
            return new Point(agent.X + agent.Width / 2, agent.Y + agent.Height / 2);
        }

        public Rectangle GetFowGeometry()
        {
            return new Rectangle(0, 0, FOW_WIDTH, FOW_HEIGHT);
        }

        /// <summary>
        /// Returns POW borders rectangle reduced by 1 pixel
        /// </summary>
        /// <returns></returns>
        public Rectangle GetPowGeometry()
        {
            Point powCentre = GetPowCenter();
            Size halfPowSize = new Size(POW_WIDTH / 2 - 1, POW_HEIGHT / 2 - 1);
            Size powSize = new Size(POW_WIDTH - 2, POW_HEIGHT - 2);
            return new Rectangle(powCentre - halfPowSize, powSize);
        }

        public Rectangle GetAgentGeometry()
        {
            return Agent.GetGeometry();
        }

        public Point RandomPositionInsideRectangle(Random rndGen, Size size, Rectangle rectangle)
        {
            return new Point(
                rndGen.Next(rectangle.Width - size.Width) + rectangle.X,
                rndGen.Next(rectangle.Height - size.Height) + rectangle.Y);
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="rndGen"></param>
        /// <param name="size"></param>
        /// <param name="minMarginFromAgent"> if -1, collision is allowed</param>
        /// <returns>Point inside POW with respect to size</returns>
        public Point RandomPositionInsidePow(Random rndGen, Size size, int minMarginFromAgent = -1)
        {
            Point randPointInPow = RandomPositionInsideRectangle(rndGen, size, GetPowGeometry());

            if (minMarginFromAgent == -1)
            {
                return randPointInPow;
            }

            Rectangle agent = GetAgentGeometry();

            Size borders = new Size(minMarginFromAgent, minMarginFromAgent);

            randPointInPow -= borders;
            size += borders + borders;

            Rectangle obj = new Rectangle(randPointInPow, size);

            Rectangle agentGeometry = Agent.GetGeometry();

            for (int i = 0; i < 1; i++)
            {
                if (m_randomPositionCounter > 1000)
                {
                    throw new Exception("Cannot place object randomly");
                }
                if (agentGeometry.IntersectsWith(obj) || obj.IntersectsWith(agentGeometry) ||
                    agent.IntersectsWith(obj) || obj.IntersectsWith(agent))
                {
                    obj.Location = RandomPositionInsideRectangle(rndGen, size, GetPowGeometry());
                    m_randomPositionCounter++;
                    i = -1; // reset cycle
                }
            }
            MyLog.Writer.WriteLine(MyLogLevel.DEBUG, "Number of unsuccessful attempts of random object placing: " + m_randomPositionCounter);
            m_randomPositionCounter = 0;

            randPointInPow = obj.Location + borders;
            size = obj.Size - borders - borders;
            obj = new Rectangle(randPointInPow, size);
            return obj.Location;
        }

        public Point RandomPositionInsidePowNonCovering(Random rndGen, Size size, int minMarginBetweenObjects = 1)
        {
            return RandomPositionInsideRectangleNonCovering(rndGen, size, this.GetPowGeometry(), minMarginBetweenObjects);
        }


        private int m_randomPositionCounter = 0;
        public Point RandomPositionInsideRectangleNonCovering(Random rndGen, Size size, Rectangle rectangle, int minMarginBetweenObjects = 1)
        {
            Point randPointInPow = RandomPositionInsideRectangle(rndGen, size, rectangle);

            Rectangle agent = GetAgentGeometry();

            Size borders = new Size(minMarginBetweenObjects, minMarginBetweenObjects);

            randPointInPow -= borders;
            size += borders + borders;

            Rectangle obj = new Rectangle(randPointInPow, size);

            for (int i = 0; i < gameObjects.Count; i++)
            {
                if (m_randomPositionCounter > 1000)
                {
                    throw new Exception("Cannot place object randomly");
                }
                Rectangle gameObjectG = gameObjects[i].GetGeometry();
                if (gameObjectG.IntersectsWith(obj) || obj.IntersectsWith(gameObjectG) ||
                    agent.IntersectsWith(obj) || obj.IntersectsWith(agent))
                {
                    obj.Location = RandomPositionInsideRectangle(rndGen, size, rectangle);
                    m_randomPositionCounter++;
                    i = -1; // reset cycle
                }
            }
            MyLog.Writer.WriteLine(MyLogLevel.DEBUG, "Number of unsuccessful attempts of random object placing: " + m_randomPositionCounter);
            m_randomPositionCounter = 0;

            randPointInPow = obj.Location + borders;
            size = obj.Size - borders - borders;
            obj = new Rectangle(randPointInPow, size);
            return obj.Location;
        }

        public GameObject CreateGameObject(Point p, GameObjectType type, string path, int width = 0, int height = 0)
        {
            GameObject rmk = new GameObject(type, path, p.X, p.Y, width, height);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Point p, Shape.Shapes shape, GameObjectType type = GameObjectType.None, int width = 0, int height = 0, float rotation = 0)
        {
            Shape rmk = new Shape(shape, p.X, p.Y, type, width, height, rotation);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Point p, Shape.Shapes shape, Color color, Size size, GameObjectType type = GameObjectType.None, float rotation = 0)
        {
            Shape rmk = new Shape(shape, p.X, p.Y, type, size.Width, size.Height, rotation);
            rmk.SetColor(color);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Point p, Shape.Shapes shape, Color color, GameObjectType type = GameObjectType.None, int width = 0, int height = 0, float rotation = 0)
        {
            Shape rmk = new Shape(shape, p.X, p.Y, type, width, height, rotation);
            rmk.SetColor(color);
            AddGameObject(rmk);
            return rmk;
        }

        private int LoadAndGetBitmapSize(string path)
        {
            if (m_bitmapTable.ContainsKey(path))
                return m_bitmapTable[path].Item1.Width * m_bitmapTable[path].Item1.Height * PIXEL_SIZE;

            foreach (string dir in new string[] { TEXTURE_DIR, TEXTURE_DIR_COMMON })
            {
                try
                {
                    string filePath = MyResources.GetMyAssemblyPath() + "\\" + dir + "\\" + path;
                    if (!File.Exists(filePath))
                        continue;
                    Bitmap bitmap = (Bitmap)Image.FromFile(filePath, true);
                    m_bitmapTable[path] = new Tuple<Bitmap, int>(bitmap, m_totalOffset);

                    if (bitmap.PixelFormat != PixelFormat.Format32bppArgb)
                    {
                        throw new ArgumentException("The specified image is not in the required RGBA format."); // note: alpha must not be premultiplied
                    }

                    return bitmap.Width * bitmap.Height * PIXEL_SIZE;
                }
                catch (Exception ex)
                {
                    m_errorMessage = ex.Message;
                }
            }

            m_errorMessage = "Could not find texture " + path;
            return 0;
        }

        private int FillWithChannelFromBitmap(Bitmap bitmap, float[] buffer, int offset)
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
            return bitmap.Width * bitmap.Height * PIXEL_SIZE;
        }

        public virtual MovableGameObject CreateAgent(string iconPath, int x = 0, int y = 0)
        {
            MovableGameObject agent = new MovableGameObject(GameObjectType.Agent, iconPath, x, y);
            //agent.CollisionResolution = CollisionResolutionType.LoseSpeed;
            AddGameObject(agent);
            Agent = agent;
            return agent;
        }

        public virtual void SetAgent(MovableGameObject agent)
        {
            //agent.CollisionResolution = CollisionResolutionType.LoseSpeed;
            AddGameObject(agent);
            Agent = agent;
        }

        public virtual void FreezeWorld(bool shouldFreeze)
        {
            m_IsWorldFrozen = shouldFreeze;
        }

        ////TODO: if two objects share the same texture, do not load it twice into memory
        public void AddGameObject(GameObject item)
        {

            if (item.bitmapPath != null)
            {

                bool isMissing = true;
                if (m_bitmapTable.ContainsKey(item.bitmapPath))
                    isMissing = false;

                int size = LoadAndGetBitmapSize(item.bitmapPath);

                Debug.Assert(size > 0, "Size of loaded bitmap is zero or negative.");
                Bitmaps.SafeCopyToDevice();
                if (isMissing)
                    Bitmaps.Reallocate(Bitmaps.Count + size);
                CudaDeviceVariable<float> devBitmaps = Bitmaps.GetDevice(this);

                Bitmap bitmap = m_bitmapTable[item.bitmapPath].Item1;

                item.bitmapPixelSize = new Size(bitmap.Width, bitmap.Height);
                if (item.Width == 0 || item.Height == 0) // object can have size independent of the texture
                {
                    item.Width = item.bitmapPixelSize.Width;
                    item.Height = item.bitmapPixelSize.Height;
                }
                item.bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * m_bitmapTable[item.bitmapPath].Item2;

                if (isMissing)
                {
                    int bitOffset = FillWithChannelFromBitmap(bitmap, Bitmaps.Host, m_totalOffset);
                    m_bitmapTable[item.bitmapPath] = new Tuple<Bitmap, int>(bitmap, m_totalOffset);
                    m_totalOffset += bitOffset;
                }

                Bitmaps.SafeCopyToDevice();
            }

            Debug.Assert(item.ArraySize >= 0, "You should not create object with negative size.");
            Objects.Reallocate(Objects.Count + item.ArraySize);

            // agent should be in front in most cases
            if (item.type == GameObjectType.Agent)
            {
                item.Layer = 10;
            }

            gameObjects.Add(item);
            gameObjects = gameObjects.OrderBy(o1 => o1.Layer).ToList();
        }

        /// <summary>
        /// Adds game object with defined layer.
        /// </summary>
        /// <param name="item"></param>
        /// <param name="layer">
        /// Layers are rendered from lowest to greatest, so greater layer cover lower.
        /// Agent is in layer 10 by default.
        /// </param>
        public void AddGameObject(GameObject item, int layer)
        {
            item.Layer = layer;
            AddGameObject(item);
        }

        public void SetGameObjectLayer(GameObject item, int layer)
        {
            item.Layer = layer;
            gameObjects = gameObjects.OrderBy(o1 => o1.Layer).ToList();
        }

        public MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            List<IMyExecutable> planCopy = new List<IMyExecutable>();

            // copy default plan content into a list for easier manipulation
            foreach (IMyExecutable task in defaultPlan.Children)
                planCopy.Add(task);

            // I know what tasks should be first:
            //IMyExecutable learningTaskStep = planCopy.Find(task => task is LearningTaskStepTask);
            IMyExecutable getInput = planCopy.Find(task => task is InputTask);
            IMyExecutable updateTask = planCopy.Find(task => task is UpdateTask);
            //IMyExecutable render = planCopy.Find(task => task is RenderTask);
            IMyExecutable render = planCopy.Find(task => task is RenderGLTask);

            HashSet<IMyExecutable> positionSensitiveTasks = new HashSet<IMyExecutable>();
            positionSensitiveTasks.Add(getInput);
            //positionSensitiveTasks.Add(learningTaskStep);
            positionSensitiveTasks.Add(updateTask);
            positionSensitiveTasks.Add(render);

            List<IMyExecutable> newPlan = new List<IMyExecutable>();

            // add tasks that have to be at the beginning of the plan, in this order:
            //newPlan.Add(learningTaskStep);
            newPlan.Add(getInput);
            newPlan.Add(updateTask);

            // add tasks that need not be at the beginning nor at the end:
            foreach (IMyExecutable task in planCopy)
            {
                if (positionSensitiveTasks.Contains(task))
                    continue;
                newPlan.Add(task);
            }

            // add tasks that have to be at the end of the plan:
            newPlan.Add(render);

            // return new plan as MyExecutionBlock
            return new MyExecutionBlock(newPlan.ToArray());
        }

        public override void Dispose()
        {
            RenderGLWorldTask.Dispose();
            base.Dispose();
        }

        public virtual InputTask GetInputTask { get; protected set; }
        public virtual UpdateTask UpdateWorldTask { get; protected set; }

        [MyTaskGroup("Rendering")]
        public RenderGLTask RenderGLWorldTask { get; protected set; }
        [MyTaskGroup("Rendering")]
        public RenderTask RenderWorldTask { get; protected set; }

        public class InputTask : MyTask<ManInWorld>
        {
            public override void Init(int nGPU) { }
            public override void Execute()
            {
                Owner.Controls.SafeCopyToHost();
                Owner.ContinuousAction = Owner.Controls.Host[0];
            }
        }

        /// <summary>
        /// Creates agent with default texture in the middle of field.
        /// </summary>
        /// <returns>Agent</returns>
        public abstract MovableGameObject CreateAgent();
        public abstract MovableGameObject CreateAgent(Point p, float size = 1.0f);


        /// <summary>
        /// Creates agenet in the centre of POW. Agents size is 0x0, he's invisible.
        /// </summary>
        /// <returns>Agent as MovableGameObject</returns>
        public virtual MovableGameObject CreateNonVisibleAgent()
        {
            MovableGameObject agent = CreateAgent(null, FOW_WIDTH / 2, FOW_HEIGHT / 2);
            Agent.IsAffectedByGravity = false;
            return agent;
        }

        public abstract GameObject CreateWall(Point p, float size = 1.0f);
        public abstract GameObject CreateTarget(Point p, float size = 1.0f);
        public abstract MovableGameObject CreateMovableTarget(Point p, float size = 1.0f);
        public abstract GameObject CreateDoor(Point p, bool isClosed = true, float size = 1.0f);
        public abstract GameObject CreateLever(Point p, bool isOn = false, float size = 1.0f);
        public abstract GameObject CreateLever(Point p, ISwitchable obj, bool isOn = false, float size = 1.0f);
        public abstract GameObject CreateRogueKiller(Point p, float size = 1.0f);
        public abstract MovableGameObject CreateRogueMovableKiller(Point p, float size = 1.0f);

        public virtual Grid GetGrid()
        {
            return new Grid(GetFowGeometry().Size, DEFAULT_GRID_SIZE);
        }

        /// <summary>
        /// For current step sets reward for the agent
        /// </summary>
        /// <param name="reward">Should be between 1 and -1</param>
        public void SetRewardForCurrentStep(float reward)
        {
            Reward.Host[0] = reward;
        }

        public void ResetReward()
        {
            Reward.Host[0] = 0;
        }

        public class UpdateTask : MyTask<ManInWorld>
        {
            public override void Init(int nGPU) { }


            public virtual void UpdatePreviousValues()
            {
                //Iterate all objects (by discarding the ones at are not Movable) and update previous X,Y values (the values of X and Y in the previous simulation step)
                for (int i = 0; i < Owner.gameObjects.Count; i++)
                {
                    GameObject obj = Owner.gameObjects[i];
                    MovableGameObject mobj = obj as MovableGameObject;

                    if (mobj == null)
                        continue;

                    mobj.previousX = obj.X;
                    mobj.previousY = obj.Y;

                    mobj.previousvX = mobj.vX;
                    mobj.previousvY = mobj.vY;
                }

                Owner.ResetReward();
            }

            private void AnimateObjects()
            {
                if (Owner.IsWorldFrozen)
                    return;

                foreach (GameObject item in Owner.gameObjects)
                {
                    IAnimated animatedItem = item as IAnimated;
                    if (animatedItem == null)
                        continue;

                    // Debug.Assert(animatedItem.AnimationEnumerator != null, "Animation enumerator is not initialized!");

                    //AnimationItem animation = animatedItem.AnimationEnumerator.Current;
                    AnimationItem animation = animatedItem.Current;

                    if (animation.condition != null && !animation.condition())
                    {
                        animatedItem.MoveNext();
                        animation = animatedItem.Current;
                    }

                    animation.StartAnimation(item, Owner);

                    if (!animation.IsStarted)
                        animation.StartAnimation(item, Owner);

                    switch (animation.type)
                    {
                        case AnimationType.Translation:
                            {
                                Debug.Assert(animation.data.Length >= 2, "Not enough data in animation data vector.");
                                item.X += (int)animation.data[0];
                                item.Y += (int)animation.data[1];
                                break;
                            }
                        default:
                        case AnimationType.None:
                            break;
                    }
                }
            }

            public virtual void MoveWorldObjects()
            {
                if (Owner.IsWorldFrozen)
                    return;

                // Compute Movement step for all objects affected by physics
                for (int i = 0; i < Owner.gameObjects.Count; i++)
                {
                    if (Owner.gameObjects[i] is MovableGameObject)
                    {
                        GameObject obj = Owner.gameObjects[i];
                        MovableGameObject mobj = obj as MovableGameObject;

                        obj.X += (int)(mobj.vX * Owner.Time);      // Apply horizontal velocity to X position
                        obj.Y += (int)(mobj.vY * Owner.Time);      // Apply vertical velocity to Y position
                    }
                }
            }

            public virtual void HandleCollisions()
            {
                // detect collisions of objects that IsMoveableByPhysics() with any other objects.
                // When a collision is detected, handle it (do nothing, bounce, stop)

                //Create boundaries of the world
                GameObject floor = new GameObject(GameObjectType.None, null, 0, Owner.FOW_HEIGHT, Owner.FOW_WIDTH, 100);

                GameObject ceiling = new GameObject(GameObjectType.None, null, 0, -100, Owner.FOW_WIDTH, 100);

                GameObject rightSide = new GameObject(GameObjectType.None, null, Owner.FOW_WIDTH, 0, 100, Owner.FOW_HEIGHT);

                GameObject leftSide = new GameObject(GameObjectType.None, null, -100, 0, 100, Owner.FOW_HEIGHT);

                for (int i = 0; i < Owner.gameObjects.Count; i++)
                {
                    GameObject obj = Owner.gameObjects[i];
                    MovableGameObject mobj = obj as MovableGameObject;
                    if (mobj == null)
                        continue;

                    mobj.onGround = false;
                }

                // the object queue length is a constant that represents how many objects are allowed to push on each other
                // before the engine stops repositioning them correctly
                for (int iObjectQueueCounter = 0; iObjectQueueCounter < 2; iObjectQueueCounter++)
                {
                    // Check if agent is colliding with any of the objects in Owner.gameObjects.
                    // If it is, adjust its postion to a position so that it doesn't collide
                    for (int i = 0; i < Owner.gameObjects.Count; i++)
                    {
                        GameObject obj = Owner.gameObjects[i];
                        MovableGameObject mobj = obj as MovableGameObject;
                        if (mobj == null)
                            continue;

                        mobj.ActualCollisions = new List<GameObject>();

                        for (int j = 0; j < Owner.gameObjects.Count; j++) // collisions with the remaining objects
                        {
                            if (i == j)
                                continue;
                            GameObject gameObj = Owner.gameObjects[j];
                            if (CheckCollision(mobj, gameObj))
                            {
                                Owner.ConflictResolver.Resolve(mobj, gameObj);
                            }
                        }

                        // collisions with world boundaries
                        if (CheckCollision(mobj, floor)) Owner.ConflictResolver.Resolve(mobj, floor);
                        if (CheckCollision(mobj, ceiling)) Owner.ConflictResolver.Resolve(mobj, ceiling);
                        if (CheckCollision(mobj, rightSide)) Owner.ConflictResolver.Resolve(mobj, rightSide);
                        if (CheckCollision(mobj, leftSide)) Owner.ConflictResolver.Resolve(mobj, leftSide);
                        //MyLog.DEBUG.WriteLine("grounded: " + PlumberOwner.onGround);
                    }
                }
            }

            public bool CheckCollision(GameObject o1, GameObject o2)
            {
                return o1.GetGeometry().IntersectsWith(o2.GetGeometry()) || o2.GetGeometry().IntersectsWith(o1.GetGeometry());
            }

            public override void Execute()
            {
                AnimateObjects();
                UpdatePreviousValues();
                MoveWorldObjects();
                HandleCollisions();
            }

            public static Point ReturnCoordinatesBetweenTwoPoints(Point p1, Point p2, float ratio)
            {
                return new Point((int)(p1.X + ratio * (p2.Y - p1.X)), (int)(p1.Y + ratio * (p2.Y - p1.Y)));
            }
        }

        /// <summary>
        /// Render the world to the visual output.
        /// </summary>
        public class RenderTask : MyTask<ManInWorld>
        {
            private MyCudaKernel m_RgbaTextureKernel;
            private MyCudaKernel m_RgbaTextureKernelNearestNeighbor;
            private MyCudaKernel m_MaskedColorKernelNearestNeighbor;
            private MyCudaKernel m_RgbaColorKernel;
            private MyCudaKernel m_RgbBackgroundKernel;
            private MyCudaKernel m_AddRgbNoiseKernel;

            public override void Init(int nGPU)
            {
                m_RgbaTextureKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaTextureKernel");
                m_RgbaTextureKernelNearestNeighbor = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaTextureKernelNearestNeighbor");
                m_MaskedColorKernelNearestNeighbor = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawMaskedColorKernelNearestNeighbor");
                m_RgbaColorKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaColorKernel");
                m_RgbBackgroundKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbBackgroundKernel");
                m_AddRgbNoiseKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "AddRgbNoiseKernel");
                // faster but information about used textures is necessary
                //m_RgbaTextureKernel2DBlock = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "DrawRgbaTextureKernel2DBlock");
            }

            public override void Execute()
            {
                // TODO: decide if it makes sense to render POW as a cut from FOW or if it is better to render FOW and POW separately
                // curently, FOW and POW are rendered separately.

                Point powCenter = Owner.GetPowCenter();
                Point worldTopLeftInPow = new Point(-powCenter.X + Owner.POW_WIDTH / 2, -powCenter.Y + Owner.POW_HEIGHT / 2);

                // POW background
                Owner.VisualPOW.Fill(DUMMY_PIXEL);
                m_RgbaColorKernel.SetupExecution(Owner.FOW_WIDTH * Owner.FOW_HEIGHT * 3);
                m_RgbaColorKernel.Run(Owner.VisualPOW, Owner.POW_WIDTH, Owner.POW_HEIGHT, worldTopLeftInPow.X, worldTopLeftInPow.Y,
                    Owner.FOW_WIDTH, Owner.FOW_HEIGHT, ((float)Owner.BackgroundColor.R) / 255.0f, ((float)Owner.BackgroundColor.G) / 255.0f, ((float)Owner.BackgroundColor.B) / 255.0f);

                int blockDimX = Owner.FOW_WIDTH;
                int gridDimZ = 1;
                if (blockDimX > 1024)
                {
                    gridDimZ = (int)Math.Ceiling(blockDimX / 1024.0);
                    blockDimX = 1024;
                }

                m_RgbBackgroundKernel.SetupExecution(new dim3(blockDimX, 1, 1), new dim3(Owner.FOW_HEIGHT, 3, gridDimZ));
                m_RgbBackgroundKernel.Run(Owner.VisualFOW, Owner.FOW_WIDTH, Owner.FOW_HEIGHT, ((float)Owner.BackgroundColor.R) / 255.0f, ((float)Owner.BackgroundColor.G) / 255.0f, ((float)Owner.BackgroundColor.B) / 255.0f);

                int offset = 0;
                for (int i = 0; i < Owner.gameObjects.Count; i++)
                {
                    GameObject g = Owner.gameObjects[i];

                    if (g.bitmapPath == null)
                        continue;

                    /*if (g.bitmapPixelSize.x == g.Width) // no texture scaling
                    {
                        m_RgbaTextureKernel.SetupExecution(g.bitmapPixelSize.Width * g.bitmapPixelSize.Height * 3);
                        //m_RgbaTextureKernel2DBlock.SetupExecution(new dim3(g.pixelSize.Width, 8, 1), new dim3(g.pixelSize.Height / 8, 3, 1));
                        m_RgbaTextureKernel.Run(Owner.Visual, Owner.FOW_WIDTH, Owner.FOW_HEIGHT, g.X, g.Y, g.bitmap, g.bitmapPixelSize.Width, g.bitmapPixelSize.Height);

                        int2 powTopLeft = new int2(powCenter.X - Owner.POW_WIDTH / 2,
                                                    powCenter.Y - Owner.POW_HEIGHT / 2);
                        m_RgbaTextureKernel.Run(Owner.AgentVisual, Owner.POW_WIDTH, Owner.POW_HEIGHT,
                            g.X - powTopLeft.X, g.Y - powTopLeft.Y, g.bitmap, g.bitmapPixelSize.Width, g.bitmapPixelSize.Height);
                    }
                    else*/
                    // texture scaling - nearest neighbor
                    {
                        if (g.isBitmapAsMask)
                        {
                            m_MaskedColorKernelNearestNeighbor.SetupExecution(g.Width * g.Height * 3);
                            m_MaskedColorKernelNearestNeighbor.Run(Owner.VisualFOW, Owner.FOW_WIDTH, Owner.FOW_HEIGHT, g.X, g.Y,
                                g.bitmap, g.bitmapPixelSize.Width, g.bitmapPixelSize.Height,
                                g.Width, g.Height, ((float)g.maskColor.B) / 255.0f, ((float)g.maskColor.G) / 255.0f, ((float)g.maskColor.R) / 255.0f);

                            Point powTopLeft = new Point(powCenter.X - Owner.POW_WIDTH / 2,
                                                        powCenter.Y - Owner.POW_HEIGHT / 2);
                            m_MaskedColorKernelNearestNeighbor.Run(Owner.VisualPOW, Owner.POW_WIDTH, Owner.POW_HEIGHT,
                                g.X - powTopLeft.X, g.Y - powTopLeft.Y, g.bitmap, g.bitmapPixelSize.Width, g.bitmapPixelSize.Height,
                                g.Width, g.Height, ((float)g.maskColor.B) / 255.0f, ((float)g.maskColor.G) / 255.0f, ((float)g.maskColor.R) / 255.0f);
                        }
                        else
                        {
                            m_RgbaTextureKernelNearestNeighbor.SetupExecution(g.Width * g.Height * 3);
                            m_RgbaTextureKernelNearestNeighbor.Run(Owner.VisualFOW, Owner.FOW_WIDTH, Owner.FOW_HEIGHT, g.X, g.Y,
                                g.bitmap, g.bitmapPixelSize.Width, g.bitmapPixelSize.Height,
                                g.Width, g.Height);

                            Point powTopLeft = new Point(powCenter.X - Owner.POW_WIDTH / 2,
                                                        powCenter.Y - Owner.POW_HEIGHT / 2);
                            m_RgbaTextureKernelNearestNeighbor.Run(Owner.VisualPOW, Owner.POW_WIDTH, Owner.POW_HEIGHT,
                                g.X - powTopLeft.X, g.Y - powTopLeft.Y, g.bitmap, g.bitmapPixelSize.Width, g.bitmapPixelSize.Height,
                                g.Width, g.Height);
                        }
                    }


                    Array.Copy(g.ToArray(), 0, Owner.Objects.Host, offset, g.ArraySize);
                    offset += g.ArraySize;
                }
                Owner.Objects.SafeCopyToDevice();

                // add noise over POW
                if (Owner.IsImageNoise)
                {
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.AgentVisualTemp.GetDevice(Owner),
                        Owner.ImageNoiseMean / 256.0f, Owner.ImageNoiseStandardDeviation / 256.0f);

                    Owner.AgentVisualTemp.SafeCopyToHost();

                    m_AddRgbNoiseKernel.SetupExecution(new dim3(Owner.POW_WIDTH, 1, 1), new dim3(Owner.POW_HEIGHT, 3, 1));
                    m_AddRgbNoiseKernel.Run(Owner.VisualPOW, Owner.POW_WIDTH, Owner.POW_HEIGHT, Owner.AgentVisualTemp.GetDevicePtr(Owner));
                }
            }
        }

        /// <summary>
        /// Render the world to the visual output.
        /// </summary>
        public class RenderGLTask : MyTask<ManInWorld>
        {
            uint m_fboHandle;
            uint m_renderTextureHandle;

            private uint m_sharedBufferHandle;
            private CudaOpenGLBufferInteropResource m_renderResource;

            private MyCudaKernel m_AddRgbNoiseKernel;

            INativeWindow m_window = null;
            IGraphicsContext m_context = null;

            private Dictionary<String, int> m_textureHandles;
            bool m_glInitialized;

            public override void Init(int nGPU)
            {
                m_AddRgbNoiseKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "AddRgbNoiseKernel");

                m_textureHandles = new Dictionary<string, int>();
                m_glInitialized = false;
            }

            public override void Execute()
            {
                if (!m_glInitialized)
                {
                    onlyOnce();
                    m_glInitialized = true;
                }

                // init textures
                UpdateTextures();

                /* FOW currently unused
                // fow
                setupFOWview();
                RenderGL();
                copyPixelsFOW();*/

                // pow
                setupPOWview();
                RenderGL();
                copyPixelsPOW();
            }

            void onlyOnce()
            {
                if (m_context != null)
                    m_context.Dispose();
                if (m_window != null)
                    m_window.Dispose();

                m_window = new NativeWindow();
                m_context = new GraphicsContext(GraphicsMode.Default, m_window.WindowInfo);
                m_context.MakeCurrent(m_window.WindowInfo);
                m_context.LoadAll();

                // Setup rendering texture
                m_renderTextureHandle = (uint)GL.GenTexture();
                GL.BindTexture(TextureTarget.Texture2D, m_renderTextureHandle);
                GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Owner.FOW_WIDTH, Owner.FOW_HEIGHT, 0, OpenTK.Graphics.OpenGL.PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);

                // Setup FBO
                m_fboHandle = (uint)GL.GenFramebuffer();
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, m_fboHandle);
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, m_renderTextureHandle, 0);


                // Setup Cuda <-> OpenGL interop
                int length = Owner.POW_HEIGHT * Owner.POW_WIDTH * sizeof(uint);
                //unbind - just in case this is causing us the invalid exception problems
                GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                //create buffer
                GL.GenBuffers(1, out m_sharedBufferHandle);
                GL.BindBuffer(BufferTarget.PixelPackBuffer, m_sharedBufferHandle);
                GL.BufferData(BufferTarget.PixelPackBuffer, (IntPtr)(length), IntPtr.Zero, BufferUsageHint.StaticRead);  // use data instead of IntPtr.Zero if needed
                GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);

                try
                {
                    m_renderResource = new CudaOpenGLBufferInteropResource(m_renderTextureHandle, CUGraphicsRegisterFlags.ReadOnly); // Read only by CUDA
                }
                catch (CudaException e)
                {
                    MyLog.INFO.WriteLine(
                        "{0}: CUDA-OpenGL interop error while itializing texture (using fallback): {1}",
                        GetType().Name, e.Message);
                }

                // Clean up
                GL.BindTexture(TextureTarget.Texture2D, 0);
                FramebufferErrorCode err = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            }

            public void UpdateTextures()
            {
                for (int i = 0; i < Owner.gameObjects.Count; i++)
                {
                    var gameObject = Owner.gameObjects[i];
                    // masks currently not supported, loading disabled
                    // shapes are drawn directly through vertices
                    if (!gameObject.isBitmapAsMask && gameObject.bitmapPath != null)
                    {
                        int loadedTextureHandle;
                        // We are assuming the gameObject.bitmapPath is the most up-to-date information about what should be rendered
                        bool loaded = m_textureHandles.TryGetValue(gameObject.bitmapPath, out loadedTextureHandle);    // returns null if not present?
                        if (!loaded)
                        {
                            // generate handle for new texture
                            GL.GenTextures(1, out loadedTextureHandle);
                            m_textureHandles.Add(gameObject.bitmapPath, loadedTextureHandle);

                            // load the bitmap for the texture here
                            GL.BindTexture(TextureTarget.Texture2D, loadedTextureHandle);
                            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                            Owner.LoadAndGetBitmapSize(gameObject.bitmapPath);
                            Bitmap bmp = Owner.m_bitmapTable[gameObject.bitmapPath].Item1;
                            BitmapData data = bmp.LockBits(
                                new Rectangle(0, 0, gameObject.bitmapPixelSize.Width, gameObject.bitmapPixelSize.Height),
                                ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, data.Width, data.Height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, data.Scan0);

                            bmp.UnlockBits(data);
                        }

                        // update texture for the gameObject
                        gameObject.SpriteTextureHandle = loadedTextureHandle;
                    }
                }
            }

            void setupPOWview()
            {
                Point powCenter = Owner.GetPowCenter();
                // Setup rendering
                GL.Viewport(0, 0, Owner.POW_WIDTH, Owner.POW_HEIGHT);

                GL.MatrixMode(MatrixMode.Projection);
                GL.LoadIdentity();
                GL.Ortho(powCenter.X - (float)Owner.POW_WIDTH / 2, powCenter.X + (float)Owner.POW_WIDTH / 2, powCenter.Y - (float)Owner.POW_HEIGHT / 2, powCenter.Y + (float)Owner.POW_HEIGHT / 2, -1, 1);
                GL.MatrixMode(MatrixMode.Modelview);
                GL.LoadIdentity();
            }

            void copyPixelsPOW()
            {
                // Prepare the results for CUDA

                // bind pixel buffer object
                GL.BindBuffer(BufferTarget.PixelPackBuffer, m_sharedBufferHandle);
                // bind buffer from which data will be read
                GL.ReadBuffer(ReadBufferMode.ColorAttachment0);
                // read data to PBO (IntPtr.Zero means offset is 0)
                GL.ReadPixels(0, 0, Owner.POW_WIDTH, Owner.POW_HEIGHT, OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedInt8888Reversed, IntPtr.Zero);
                GL.ReadBuffer(ReadBufferMode.None);

                if (m_renderResource != null && m_renderResource.IsRegistered && !m_renderResource.IsMapped)
                {
                    // map the interop resource
                    m_renderResource.Map();
                }

                int size = Owner.POW_HEIGHT * Owner.POW_WIDTH * Marshal.SizeOf(typeof(uint));
                Owner.VisualPOW.GetDevice(Owner).CopyToDevice(m_renderResource.GetMappedPointer<uint>().DevicePointer, 0, 0, size);

                // deinit CUDA interop
                m_renderResource.UnMap();
                GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);

                // add noise over POW
                if (Owner.IsImageNoise)
                {
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal32(Owner.AgentVisualTemp.GetDevice(Owner).DevicePointer, (SizeT)(Owner.AgentVisualTemp.Count), Owner.ImageNoiseMean, Owner.ImageNoiseStandardDeviation);

                    m_AddRgbNoiseKernel.SetupExecution(Owner.POW_HEIGHT * Owner.POW_WIDTH);
                    m_AddRgbNoiseKernel.Run(Owner.VisualPOW, Owner.POW_WIDTH, Owner.POW_HEIGHT, Owner.AgentVisualTemp);
                }
            }

            void RenderGL()
            {
                // maybe unnecessary fix of the bug with garbage collected old m_windows:
                //m_window.ProcessEvents();

                m_context.MakeCurrent(m_window.WindowInfo);
                GL.Finish();

                GL.BindFramebuffer(FramebufferTarget.Framebuffer, m_fboHandle);

                //GL.PushAttrib(AttribMask.ViewportBit); // stores GL.Viewport() parameters


                // For POW
                //GL.LoadMatrix(Matrix4.LookAt(...));

                GL.ClearColor(Owner.BackgroundColor);

                GL.Enable(EnableCap.Texture2D);
                GL.Enable(EnableCap.Blend);

                GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

                GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

                GL.End();

                // Render game objects
                // TODO: object rendering order -- environment first, then creatures and active objects
                foreach (var gameObject in Owner.gameObjects)
                {

                    GL.PushMatrix();

                    // translate object to its position in the scene
                    GL.Translate((float)gameObject.X, (float)gameObject.Y, 0.0f);

                    GL.Scale((float)gameObject.Width, (float)gameObject.Height, 1f);

                    // translate back
                    GL.Translate(0.5f, 0.5f, 0.0f);

                    // rotate around center (origin)
                    GL.Rotate(gameObject.Rotation, 0.0f, 0.0f, 1.0f);

                    // translate s.t. object center in origin
                    GL.Translate(-0.5f, -0.5f, 0.0f);

                    if (gameObject.isBitmapAsMask)
                    {
                        // gameObject is a shape -> draw it directly
                        //((Shape)gameObject).ShapeType = Shape.Shapes.Triangle;
                        GL.BindTexture(TextureTarget.Texture2D, m_renderTextureHandle);
                        drawShape(gameObject);
                    }
                    else if (gameObject.bitmapPath != null)
                    {
                        // gameObject has a texture -> draw it
                        GL.BindTexture(TextureTarget.Texture2D, gameObject.SpriteTextureHandle);
                        GL.Begin(PrimitiveType.Quads);
                        GL.TexCoord2(0.0f, 0.0f); GL.Vertex2(0f, 0f);
                        GL.TexCoord2(1.0f, 0.0f); GL.Vertex2(1f, 0f);
                        GL.TexCoord2(1.0f, 1.0f); GL.Vertex2(1f, 1f);
                        GL.TexCoord2(0.0f, 1.0f); GL.Vertex2(0f, 1f);
                        GL.End();
                    }

                    GL.PopMatrix();
                }

                // Clean up
                GL.BindTexture(TextureTarget.Texture2D, 0);
                GL.Disable(EnableCap.Texture2D);
                GL.Disable(EnableCap.Blend);

                //GL.PopAttrib(); // restores GL.Viewport() parameters
            }

            public void drawShape(GameObject gameObject)
            {
                Shape s = (Shape)gameObject;

                GL.Color4(gameObject.maskColor);
                GL.Begin(PrimitiveType.Polygon);

                switch (s.ShapeType)
                {
                    case Shape.Shapes.Circle:
                        drawCircle();
                        break;
                    case Shape.Shapes.Square:
                        drawSquare();
                        break;
                    case Shape.Shapes.Triangle:
                        drawTriangle();
                        break;
                    case Shape.Shapes.Star:
                        drawStar();
                        break;
                    case Shape.Shapes.Pentagon:
                        drawPentagon();
                        break;
                    case Shape.Shapes.Mountains:
                        drawMountains();
                        break;
                    case Shape.Shapes.T:
                        drawT();
                        break;
                    case Shape.Shapes.Tent:
                        drawTent();
                        break;
                    case Shape.Shapes.Rhombus:
                        drawRhombus();
                        break;
                    case Shape.Shapes.DoubleRhombus:
                        drawDoubleRhombus();
                        break;
                    default:
                        // reset color
                        GL.Color4(Color.White);
                        throw new ArgumentException("Unknown shape");
                }
                GL.End();
                // reset color
                GL.Color4(Color.White);
            }

            void drawTriangle()
            {
                GL.Vertex2(0, 0);
                GL.Vertex2(1, 0);
                GL.Vertex2(0.5f, 0.707f);  // 0.5, 1/sqrt(2)
            }

            void drawSquare()
            {
                GL.Vertex2(0f, 0f);
                GL.Vertex2(1f, 0f);
                GL.Vertex2(1f, 1f);
                GL.Vertex2(0f, 1f);
            }

            void drawCircle()
            {
                float deg2rad = 3.14159f / 180;
                for (int i = 0; i < 360; i++)
                {
                    float degInRad = i * deg2rad;
                    GL.Vertex2((Math.Cos(degInRad) + 1) / 2, (Math.Sin(degInRad) + 1) / 2);
                }
            }

            void drawPentagon()
            {
                GL.Vertex2(1.0, 0.5);
                GL.Vertex2(0.654507120060765305, 0.97552870560096394);
                GL.Vertex2(0.095489800597887, 0.793890283234513885);
                GL.Vertex2(0.095489800597887, 0.206109716765486115);
                GL.Vertex2(0.654514005681348905, 0.024473531705853315);
            }

            void drawStar()
            {
                GL.Vertex2(0.5f, 0.26f);
                GL.Vertex2(0.82f, 0.0f);
                GL.Vertex2(0.75f, 0.38f);
                GL.Vertex2(1f, 0.6f);
                GL.Vertex2(0.66f, 0.62f);
                GL.Vertex2(0.5f, 1f);
                GL.Vertex2(0.34f, 0.62f);
                GL.Vertex2(0f, 0.6f);
                GL.Vertex2(0.25f, 0.38f);
                GL.Vertex2(0.18, 0.0);
            }

            void drawMountains()
            {
                GL.Vertex2(0.5f, 0.5f);
                GL.Vertex2(0.66f, 0f);
                GL.Vertex2(1f, 1f);
                GL.Vertex2(0f, 1);
                GL.Vertex2(0.33f, 0f);
            }

            void drawT()
            {
                GL.Vertex2(0.6f, 0.2f);
                GL.Vertex2(0.6f, 1f);
                GL.Vertex2(0.4f, 1f);
                GL.Vertex2(0.4f, 0.2f);
                GL.Vertex2(0f, 0.2f);
                GL.Vertex2(0f, 0f);
                GL.Vertex2(1f, 0f);
                GL.Vertex2(1f, 0.2f);
            }

            void drawTent()
            {
                GL.Vertex2(0.5f, 0f);
                GL.Vertex2(1f, 0.5f);
                GL.Vertex2(1f, 1f);
                GL.Vertex2(0.5f, 0.5f);
                GL.Vertex2(0f, 1f);
                GL.Vertex2(0f, 0.5f);
            }

            void drawRhombus()
            {
                GL.Vertex2(0.33f, 0f);
                GL.Vertex2(1f, 0f);
                GL.Vertex2(0.66f, 1f);
                GL.Vertex2(0f, 1f);
            }

            void drawDoubleRhombus()
            {
                GL.Vertex2(0.5f, 0.4f);
                GL.Vertex2(0.75f, 0f);
                GL.Vertex2(1f, 0.5f);
                GL.Vertex2(0.75f, 1f);
                GL.Vertex2(0.5f, 0.6f);
                GL.Vertex2(0.25f, 1f);
                GL.Vertex2(0f, 0.5f);
                GL.Vertex2(0.25f, 0f);
            }

            internal void Dispose()
            {
                if (m_context != null)
                    m_context.Dispose();
                if (m_window != null)
                    m_window.Dispose();

                m_window = new NativeWindow();
                m_context = new GraphicsContext(GraphicsMode.Default, m_window.WindowInfo);
                m_context.MakeCurrent(m_window.WindowInfo);
                m_context.LoadAll();

                GL.BindTexture(TextureTarget.Texture2D, 0);
                // delete textures
                if (m_textureHandles != null)
                {
                    foreach (int handle in m_textureHandles.Values)
                    {
                        int h = handle;
                        GL.DeleteTextures(1, ref h);
                    }
                }

                if (m_renderTextureHandle != 0)
                {
                    GL.DeleteTextures(1, ref m_renderTextureHandle);
                }

                // delete FbO
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                if (m_fboHandle != 0)
                {
                    GL.DeleteFramebuffers(1, ref m_fboHandle);
                }

                // delete PBO
                GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                if (m_sharedBufferHandle != 0)
                {
                    GL.DeleteBuffers(1, ref m_sharedBufferHandle);
                }

                // delete CUDA <-> GL interop
                if (m_renderResource.IsMapped)
                {
                    m_renderResource.UnMap();
                }
                if (m_renderResource.IsRegistered)
                {
                    m_renderResource.Unregister();
                }
                m_renderResource.Dispose();

                if (m_context != null)
                {
                    m_context.Dispose();
                }
                if (m_window != null)
                {
                    m_window.Dispose();
                }
            }
        }
    }
}
