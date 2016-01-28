using GoodAI.Core;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

namespace GoodAI.Modules.School.Worlds
{
    /// <author>GoodAI</author>
    /// <meta>Mp,Mv,Os</meta>
    /// <status>WIP</status>
    /// <summary> Implementation of a configurable 2D world </summary>
    /// <description>
    /// Implementation of a configurable 2D world
    /// </description>
    public abstract class ManInWorld : AbstractSchoolWorld, IWorldAdapter
    {
        public enum TextureSet
        {
            A = 0,
            B = 1,
            C = 2,
            D = 3,
            E = 4,
            Plumber = 5
        };

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

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Difficulty
        {
            get { return GetInput(2); }
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

        public override void SetHint(string attr, float value)
        {
            switch (attr)
            {
                case TSHintAttributes.NOISE:
                    IsImageNoise = value > 0;
                    break;
                case TSHintAttributes.DEGREES_OF_FREEDOM:
                    DegreesOfFreedom = (int)value;
                    break;
            }
        }

        public override void UpdateMemoryBlocks()
        {
            VisualFOW.Count = FOW_WIDTH * FOW_HEIGHT * 3;
            VisualFOW.ColumnHint = FOW_WIDTH;

            VisualPOW.Count = POW_HEIGHT * POW_WIDTH * 3;
            VisualPOW.ColumnHint = POW_WIDTH;

            AgentVisualTemp.Count = VisualPOW.Count;

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
            return new Rectangle(0 ,0 ,FOW_WIDTH, FOW_HEIGHT);
        }

        public Rectangle GetPowGeometry()
        {
            Point powCentre = GetPowCenter();
            Size halfPow = new Size(POW_WIDTH / 2, POW_HEIGHT / 2);
            Size pow = new Size(POW_WIDTH, POW_HEIGHT);
            return new Rectangle(powCentre - halfPow, pow);
        }

        public Rectangle GetAgentGeometry()
        {
            return Agent.GetGeometry();
        }

        // returns upper left corner
        public Point GetRandomPositionInsidePow(Random rndGen, Size s) {
            Rectangle pow = GetPowGeometry();
            Point randPointInPow= new Point(
                rndGen.Next(pow.Width - s.Width) + pow.X,
                rndGen.Next(pow.Height - s.Height) + pow.Y);

            Rectangle agent = GetAgentGeometry();

            Rectangle obj = new Rectangle(randPointInPow, s);

            while (agent.IntersectsWith(obj) || obj.IntersectsWith(agent))
            {
                obj.Location += new Size(4, 4); ;
            }
            return obj.Location;
        }

        // like GetRandomPositionInsidePow, but avoids all added GameOjects
        public Point GetRandomPositionInsidePowNonCovering(Random rndGen, Size s)
        {
            return GetRandomPositionInsideRectangleNonCovering(rndGen, s, this.GetPowGeometry());
        }

        public Point GetRandomPositionInsideRectangleNonCovering(Random rndGen, Size s, Rectangle r)
        {
            Point randPointInPow = new Point(
                rndGen.Next(r.Width - s.Width) + r.X,
                rndGen.Next(r.Height - s.Height) + r.Y);

            Rectangle agent = GetAgentGeometry();

            Rectangle obj = new Rectangle(randPointInPow, s);

            foreach (GameObject gameObject in gameObjects)
            {
                Rectangle gameObjectG = gameObject.GetGeometry();
                while (gameObjectG.IntersectsWith(obj) || obj.IntersectsWith(gameObjectG) ||
                    agent.IntersectsWith(obj) || obj.IntersectsWith(agent))
                {
                    obj.Location += new Size(4, 4); ;
                }
            }
            return obj.Location;
        }

        public GameObject CreateGameObject(Point p, GameObjectType type, string path, int width = 0, int height = 0)
        {
            GameObject rmk = new GameObject(type, path, p.X, p.Y, width, height);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Point p, Shape.Shapes shape, GameObjectType type = GameObjectType.None, int width = 0, int height = 0)
        {
            Shape rmk = new Shape(shape, p.X, p.Y, type, width, height);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Point p, Shape.Shapes shape, Color color, Size size, GameObjectType type = GameObjectType.None)
        {
            Shape rmk = new Shape(shape, p.X, p.Y, type, size.Width, size.Height);
            rmk.SetColor(color);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Point p, Shape.Shapes shape, Color color, GameObjectType type = GameObjectType.None, int width = 0, int height = 0)
        {
            Shape rmk = new Shape(shape, p.X, p.Y, type, width, height);
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

        public override void ClearWorld()
        {
            Agent = null;
            gameObjects.Clear();
            Objects.Count = 0;
            IsImageNoise = false;
            m_IsWorldFrozen = false;
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

            gameObjects.Add(item);
        }

        public override MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            List<IMyExecutable> planCopy = new List<IMyExecutable>();

            // copy default plan content into a list for easier manipulation
            foreach (IMyExecutable task in defaultPlan.Children)
                planCopy.Add(task);

            // I know what tasks should be first:
            IMyExecutable learningTaskStep = planCopy.Find(task => task is LearningTaskStepTask);
            IMyExecutable getInput = planCopy.Find(task => task is InputTask);
            IMyExecutable updateTask = planCopy.Find(task => task is UpdateTask);
            IMyExecutable render = planCopy.Find(task => task is RenderTask);

            HashSet<IMyExecutable> positionSensitiveTasks = new HashSet<IMyExecutable>();
            positionSensitiveTasks.Add(getInput);
            positionSensitiveTasks.Add(learningTaskStep);
            positionSensitiveTasks.Add(updateTask);
            positionSensitiveTasks.Add(render);

            List<IMyExecutable> newPlan = new List<IMyExecutable>();

            // add tasks that have to be at the beginning of the plan, in this order:
            newPlan.Add(learningTaskStep);
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

        public virtual InputTask GetInputTask { get; protected set; }
        public virtual UpdateTask UpdateWorldTask { get; protected set; }
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
        public abstract GameObject CreateLever(Point p, float size = 1.0f);
        public abstract GameObject CreateLever(Point p, ISwitchable obj, float size = 1.0f);
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
    }
}
