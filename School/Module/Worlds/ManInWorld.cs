using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using YAXLib;
using PixelFormat = System.Drawing.Imaging.PixelFormat;

namespace GoodAI.Modules.School.Worlds
{
    /// <author>GoodAI</author>
    /// <meta>Mp,Mv,Os,Ph,Mm</meta>
    /// <status>WIP</status>
    /// <summary> Implementation of a configurable 2D world </summary>
    /// <description>
    /// Implementation of a configurable 2D world
    /// </description>
    [DisplayName("2D worlds")]
    public abstract class ManInWorld : MyWorld, IWorldAdapter
    {
        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Controls
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0), DynamicBlock]
        public MyMemoryBlock<float> VisualFOW
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1), MyUnmanaged]
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

        [DynamicBlock]
        public MyMemoryBlock<float> Bitmaps { get; protected set; }
        public MyMemoryBlock<float> AgentVisualTemp { get; protected set; } // used e.g. for holding random numbers during noise generation
        private MyMemoryBlock<float> ControlsAdapterTemp { get; set; }

        #endregion

        #region Rendering fields

        protected virtual string TEXTURE_DIR { get { return @"res\FILL_IN_INHERITED_CLASS"; } }
        protected string TEXTURE_DIR_COMMON = @"res\SchoolWorldCommon\";

        // tuple contains the Bitmap itself and its offset in Bitmaps memory block
        private readonly Dictionary<string, Tuple<Bitmap, int>> m_bitmapTable;
        private int m_totalOffset;

        // Resolutions
        public Size Fow { get; protected set; }
        public Size Pow { get; protected set; }

        public Color BackgroundColor { get; set; }
        public bool UseBackgroundTexture { get; set; }

        public const float DUMMY_PIXEL = float.NaN;
        private const int PIXEL_SIZE = 4; // RGBA: 4 floats per pixel

        // R/G/B intensity value range is 0-256
        public float ImageNoiseStandardDeviation = 20.0f; // the noise follows a normal distribution (maybe can be simpler?)
        public float ImageNoiseMean = 0; // the average value that is added to each pixel in the image
        public bool IsImageNoise { get; set; }

        #endregion

        #region Fields

        public static Size DEFAULT_GRID_SIZE = new Size(32, 32);

        public MovableGameObject Agent { get; protected set; }
        // object which should be implemented with same actions and same behavior as agent have
        public AbstractTeacherInWorld Teacher { get; protected set; }

        public List<GameObject> GameObjects { get; private set; }
        private readonly StandardConflictResolver m_conflictResolver;

        // Game-space sizes
        public SizeF Scene { get; protected set; }
        public SizeF Viewport { get; protected set; }

        public bool IsWorldFrozen { get; set; } // nothing moves in a frozen world
        public int DegreesOfFreedom { get; set; }
        public float ContinuousAction { get; set; }

        public float Time = 1f;

        #endregion

        public ManInWorld()
        {
            Scene = Fow = new Size(1024, 1024);
            Viewport = Pow = new Size(256, 256);

            BackgroundColor = Color.FromArgb(77, 174, 255);
            m_bitmapTable = new Dictionary<string, Tuple<Bitmap, int>>();

            GameObjects = new List<GameObject>();
            m_conflictResolver = new StandardConflictResolver();

            DegreesOfFreedom = 2;
        }

        #region MyNode overrides

        public override MyMemoryBlock<float> GetInput(int index)
        {
            if (ControlsAdapterTemp != null) //HACK which checks if World is standalone or in SchoolWorld. TODO fix it somehow.
                return ControlsAdapterTemp;
            return base.GetInput<float>(index);
        }

        public override MyMemoryBlock<T> GetInput<T>(int index)
        {
            if (ControlsAdapterTemp != null) //HACK which checks if World is standalone or in SchoolWorld. TODO fix it somehow.
                return ControlsAdapterTemp as MyMemoryBlock<T>;
            return base.GetInput<T>(index);
        }

        public override MyAbstractMemoryBlock GetAbstractInput(int index)
        {
            if (ControlsAdapterTemp != null) //HACK which checks if World is standalone or in SchoolWorld. TODO fix it somehow.
                return ControlsAdapterTemp;
            return base.GetAbstractInput(index);
        }

        public override void UpdateMemoryBlocks()
        {
            if (School != null)
            {
                Pow = new Size(School.Visual.Dims[0], School.Visual.Dims[1]);
                Fow = new Size(Math.Max(Fow.Width, Pow.Width), Math.Max(Fow.Height, Pow.Height)); // TODO: does not really make sense, make it independent
            }

            VisualPOW.Dims = new TensorDimensions(Pow.Width, Pow.Height);
            VisualFOW.Dims = new TensorDimensions(Fow.Width, Fow.Height);

            AgentVisualTemp.Count = VisualPOW.Count * 3;
            Bitmaps.Count = 0;

            Objects.Count = 0;
            Reward.Count = 1;

            GameObjects.Clear();
            m_bitmapTable.Clear();

            m_totalOffset = 0;

            UseBackgroundTexture = false;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            validator.AssertError(Pow.Width <= Fow.Width, this, "PowWidth cannot be higher than FowWidth, corresponding sizes are: " + Pow.Width + ", " + Fow.Width);
            validator.AssertError(Pow.Height <= Fow.Height, this, "PowHeight cannot be higher than FowHeight, corresponding sizes are: " + Pow.Height + ", " + Fow.Height);
        }

        public override void Cleanup()
        {
            Dispose();
            base.Cleanup();
        }

        public override void Dispose()
        {
            RenderGLWorldTask.Dispose();
            base.Dispose();
        }

        #endregion

        #region IWorldAdapter overrides

        public SchoolWorld School { get; set; }
        public MyWorkingNode World { get { return this; } }

        public MyTask GetWorldRenderTask()
        {
            return RenderGLWorldTask;
        }

        public void InitAdapterMemory()
        {
            ControlsAdapterTemp = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
            ControlsAdapterTemp.Count = 128;

            School.AspectRatio = Viewport.Width / Viewport.Height;
        }


        public virtual void InitWorldInputs(int nGPU)
        { }

        public virtual void MapWorldInputs()
        {
            // Copy data from wrapper to world (inputs) - SchoolWorld validation ensures that we have something connected
            if (School.ActionInput.Owner is DeviceInput)
            {
                School.ActionInput.SafeCopyToDevice();
                ControlsAdapterTemp.Host[0] = School.ActionInput.Host[68]; // D 
                ControlsAdapterTemp.Host[1] = School.ActionInput.Host[65]; // A
                ControlsAdapterTemp.Host[2] = School.ActionInput.Host[83]; // S
                ControlsAdapterTemp.Host[3] = School.ActionInput.Host[87]; // W
                ControlsAdapterTemp.SafeCopyToDevice();
            }
            else
            {
                ControlsAdapterTemp.CopyFromMemoryBlock(School.ActionInput, 0, 0, Math.Min(ControlsAdapterTemp.Count, School.ActionInput.Count));
            }
        }

        public virtual void InitWorldOutputs(int nGPU)
        { }

        public virtual void MapWorldOutputs()
        {
            // Copy data from world to wrapper
            VisualPOW.CopyToMemoryBlock(School.Visual, 0, 0, Math.Min(VisualPOW.Count, School.Visual.Count));
            if (Objects.Count > 0)
                Objects.CopyToMemoryBlock(School.Data, 0, 0, Math.Min(Objects.Count, School.DataSize));
            //schoolWorld.Visual.Dims = VisualPOW.Dims;
            School.DataLength.Fill(Math.Min(Objects.Count, School.DataSize));
            Reward.CopyToMemoryBlock(School.RewardMB, 0, 0, 1);
        }

        public virtual void ClearWorld()
        {
            Agent = null;
            GameObjects.Clear();
            Objects.Count = 0;
            IsImageNoise = false;
            IsWorldFrozen = false;
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
            else if (attr == TSHintAttributes.IMAGE_TEXTURE_BACKGROUND)
            {
                UseBackgroundTexture = value > 0;
            }
        }

        #endregion

        #region Rendering helpers

        protected PointF GetPowCenter()
        {
            GameObject agent = Agent;
            if (agent == null)
            {
                return new PointF(Viewport.Width / 2, Viewport.Height / 2);
            }
            return new PointF(agent.Position.X + agent.Size.Width / 2, agent.Position.Y + agent.Size.Height / 2);
        }

        public RectangleF GetFowGeometry()
        {
            return new RectangleF(new PointF(), Scene);
        }

        /// <summary>
        /// Returns POW borders rectangle reduced by 1 pixel
        /// </summary>
        /// <returns></returns>
        public RectangleF GetPowGeometry()
        {
            SizeF halfPowSize = new SizeF(Pow.Width / 2, Pow.Height / 2);
            return new RectangleF(GetPowCenter() - halfPowSize, Viewport);
        }

        public RectangleF GetAgentGeometry()
        {
            return Agent.GetGeometry();
        }


        public PointF RandomPositionInsideRectangle(Random rndGen, SizeF size, RectangleF rectangle)
        {
            return new PointF(
                rndGen.Next() * (rectangle.Width - size.Width) + rectangle.X,
                rndGen.Next() * (rectangle.Height - size.Height) + rectangle.Y);
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="rndGen"></param>
        /// <param name="size"></param>
        /// <param name="agentMargin"> if -1, collision is allowed</param>
        /// <param name="objectMargin"></param>
        /// <returns>Point inside POW with respect to size</returns>
        public PointF RandomPositionInsideViewport(Random rndGen, SizeF size, float agentMargin = -1, float objectMargin = 1)
        {
            RectangleF pow = GetPowGeometry();

            SizeF objectBorders = new SizeF(objectMargin, objectMargin);
            size += objectBorders + objectBorders;

            if (agentMargin == -1)
            {
                return RandomPositionInsideRectangle(rndGen, size, pow);
            }

            RectangleF agent = GetAgentGeometry();
            SizeF agentBorders = new SizeF(agentMargin, agentMargin);
            RectangleF agentGeometry = Agent.GetGeometry();
            agentGeometry.Location -= agentBorders;
            agentGeometry.Size += agentBorders + agentBorders;

            RectangleF obj = new Rectangle();
            int randomPositionCounter = 0;
            bool intersects = true;
            while (intersects)
            {
                intersects = false;
                obj = new RectangleF(RandomPositionInsideRectangle(rndGen, size, pow), size);
                randomPositionCounter++;

                if (randomPositionCounter > 1000)
                {
                    throw new Exception("Cannot place object randomly");
                }

                if (agentGeometry.IntersectsWith(obj) || obj.IntersectsWith(agentGeometry) ||
                    agent.IntersectsWith(obj) || obj.IntersectsWith(agent))
                {
                    intersects = true;
                }
            }
            MyLog.DEBUG.WriteLine("Number of unsuccessful attempts of random object placing: " + randomPositionCounter);

            PointF randPoint = obj.Location + objectBorders;

            return randPoint;
        }

        public PointF RandomPositionInsidePowNonCovering(Random rndGen, SizeF size, int objectMargin = 1)
        {
            return RandomPositionInsideRectangleNonCovering(rndGen, size, GetPowGeometry(), objectMargin);
        }

        public PointF RandomPositionInsideRectangleNonCovering(Random rndGen, SizeF size, RectangleF rectangle, float objectMargin = 1, float agentMargin = 0)
        {
            RectangleF agent = GetAgentGeometry();
            agent = LearningTaskHelpers.ResizeRectangleAroundCentre(agent, agentMargin, agentMargin);


            SizeF borders = new SizeF(objectMargin, objectMargin);
            size += borders + borders;

            int randomPositionCounter = 0;

            RectangleF obj = new RectangleF();
            bool intersects = true;
            while (intersects)
            {
                intersects = false;
                randomPositionCounter++;
                obj = new RectangleF(RandomPositionInsideRectangle(rndGen, size, rectangle), size);

                // check intersection for all GameObjects
                for (int i = 0; i < GameObjects.Count; i++)
                {
                    if (randomPositionCounter > 1000)
                    {
                        throw new Exception("Cannot place object randomly");
                    }
                    RectangleF gameObjectG = GameObjects[i].GetGeometry();
                    if (gameObjectG.IntersectsWith(obj) ||
                        obj.IntersectsWith(gameObjectG) ||
                        agent.IntersectsWith(obj) ||
                        obj.IntersectsWith(agent))
                    {
                        intersects = true;
                        break;
                    }
                }
            }
            MyLog.DEBUG.WriteLine("Number of unsuccessful attempts of random object placing: " + randomPositionCounter);

            PointF randPoint = obj.Location + borders;
            return randPoint;
        }


        public GameObject CreateGameObject(string bitmapPath, PointF p, SizeF size = default(SizeF), GameObjectType type = GameObjectType.None)
        {
            GameObject rmk = new GameObject(bitmapPath, p, size, type);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Shape.Shapes shape, PointF p, SizeF size = default(SizeF), float rotation = 0, GameObjectType type = GameObjectType.None)
        {
            Shape rmk = new Shape(shape, p, size, rotation, type);
            AddGameObject(rmk);
            return rmk;
        }

        public GameObject CreateShape(Shape.Shapes shape, Color color, PointF p, SizeF size = default(SizeF), float rotation = 0, GameObjectType type = GameObjectType.None)
        {
            Shape rmk = new Shape(shape, p, size, rotation, type) { ColorMask = color };
            AddGameObject(rmk);
            return rmk;
        }

        private int LoadAndGetBitmapSize(string path)
        {
            if (m_bitmapTable.ContainsKey(path))
                return m_bitmapTable[path].Item1.Width * m_bitmapTable[path].Item1.Height * PIXEL_SIZE;

            foreach (string dir in new[] { TEXTURE_DIR, TEXTURE_DIR_COMMON })
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
                    MyLog.ERROR.WriteLine(ex.Message);
                }
            }

            MyLog.ERROR.WriteLine("Could not find texture " + path);
            return 0;
        }

        private int FillWithChannelFromBitmap(Bitmap bitmap, float[] buffer, int offset)
        {
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

            byte[] pixels = new byte[bitmapData.Stride];

            int bytesPerPixel = bitmapData.Stride / bitmapData.Width;
            Debug.Assert(bytesPerPixel == 4); // we expect a 32-bit ARGB Bitmap

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

        public virtual MovableGameObject CreateAgent(string iconPath, PointF position = default(PointF))
        {
            MovableGameObject agent = new MovableGameObject(iconPath, position, type: GameObjectType.Agent);
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

        /// <summary>
        /// Adds game object with defined layer.
        /// </summary>
        /// <param name="item"></param>
        /// <param name="layer">
        /// Layers are rendered from lowest to greatest, so greater layer cover lower.
        /// Agent is in layer 10 by default.
        /// </param>
        public void AddGameObject(GameObject item, int layer = -1)
        {
            if (layer >= 0)
                item.Layer = layer;

            ////TODO: if two objects share the same texture, do not load it twice into memory

            if (item.BitmapPath != null)
            {
                bool isMissing = !m_bitmapTable.ContainsKey(item.BitmapPath);
                int size = LoadAndGetBitmapSize(item.BitmapPath);

                Debug.Assert(size > 0, "Size of loaded Bitmap is zero or negative.");
                Bitmaps.SafeCopyToDevice();
                if (isMissing)
                    Bitmaps.Reallocate(Bitmaps.Count + size);
                CudaDeviceVariable<float> devBitmaps = Bitmaps.GetDevice(this);

                Bitmap bitmap = m_bitmapTable[item.BitmapPath].Item1;

                item.BitmapPixelSize = new Size(bitmap.Width, bitmap.Height);
                if (item.Size.Width == 0 || item.Size.Height == 0) // object can have size independent of the texture
                {
                    item.Size.Width = item.BitmapPixelSize.Width;
                    item.Size.Height = item.BitmapPixelSize.Height;
                }
                item.BitmapPtr = devBitmaps.DevicePointer + devBitmaps.TypeSize * m_bitmapTable[item.BitmapPath].Item2;

                if (isMissing)
                {
                    int bitOffset = FillWithChannelFromBitmap(bitmap, Bitmaps.Host, m_totalOffset);
                    m_bitmapTable[item.BitmapPath] = new Tuple<Bitmap, int>(bitmap, m_totalOffset);
                    m_totalOffset += bitOffset;
                }

                Bitmaps.SafeCopyToDevice();
            }

            Debug.Assert(item.ArraySize >= 0, "You should not create object with negative size.");
            Objects.Reallocate(Objects.Count + item.ArraySize);

            // agent should be in front in most cases
            if (item.Type == GameObjectType.Agent)
            {
                item.Layer = 10;
            }

            GameObjects.Add(item);
            GameObjects = GameObjects.OrderBy(o1 => o1.Layer).ToList();
        }

        #endregion

        #region Tasks

        public InputTask GetInputTask { get; protected set; }
        public UpdateTask UpdateWorldTask { get; protected set; }
        public RenderGLTask RenderGLWorldTask { get; protected set; }


        public class InputTask : MyTask<ManInWorld>
        {
            public override void Init(int nGPU) { }
            public override void Execute()
            {
                Owner.Controls.SafeCopyToHost();
                Owner.ContinuousAction = Owner.Controls.Host[0];
            }
        }

        #region Updating

        /// <summary>
        /// Creates agent with default texture in the middle of field.
        /// </summary>
        /// <returns>Agent</returns>
        public abstract MovableGameObject CreateAgent();
        public abstract MovableGameObject CreateAgent(PointF p, float size = 1.0f);


        /// <summary>
        /// Creates agenet in the centre of POW. Agents size is 0x0, he's invisible.
        /// </summary>
        /// <returns>Agent as MovableGameObject</returns>
        public virtual MovableGameObject CreateNonVisibleAgent()
        {
            MovableGameObject agent = CreateAgent(null, new PointF(Scene.Width / 2, Scene.Height / 2));
            Agent.IsAffectedByGravity = false;
            return agent;
        }

        public abstract GameObject CreateWall(PointF p, float size = 1.0f);
        public abstract GameObject CreateTarget(PointF p, float size = 1.0f);
        public abstract MovableGameObject CreateMovableTarget(PointF p, float size = 1.0f);
        public abstract GameObject CreateDoor(PointF p, bool isClosed = true, float size = 1.0f);
        public abstract GameObject CreateLever(PointF p, bool isOn = false, float size = 1.0f);
        public abstract GameObject CreateLever(PointF p, ISwitchable obj, bool isOn = false, float size = 1.0f);
        public abstract GameObject CreateRogueKiller(PointF p, float size = 1.0f);
        public abstract MovableGameObject CreateRogueMovableKiller(PointF p, float size = 1.0f);

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
            GameObject m_floor, m_rightSide, m_ceiling, m_leftSide;


            public override void Init(int nGPU)
            {
                //Create boundaries of the world
                PointF position = new PointF(0, Owner.Scene.Height);
                SizeF size = new SizeF(Owner.Scene.Width, 100);
                m_floor = new GameObject(null, position, size);
                position.Y = -100;
                m_ceiling = new GameObject(null, position, size);

                position = new PointF(Owner.Scene.Width, 0);
                size = new SizeF(100, Owner.Scene.Height);
                m_rightSide = new GameObject(null, position, size);
                position.X = -100;
                m_leftSide = new GameObject(null, position, size);
            }

            public virtual void UpdatePreviousValues()
            {
                //Iterate all objects (by discarding the ones at are not Movable) and update previous X,Y values (the values of X and Y in the previous simulation step)
                for (int i = 0; i < Owner.GameObjects.Count; i++)
                {
                    GameObject obj = Owner.GameObjects[i];
                    MovableGameObject mobj = obj as MovableGameObject;

                    if (mobj == null)
                        continue;

                    mobj.PositionPrevious.X = obj.Position.X;
                    mobj.PositionPrevious.Y = obj.Position.Y;

                    mobj.VelocityPrevious.X = mobj.Velocity.X;
                    mobj.VelocityPrevious.Y = mobj.Velocity.Y;
                }

                Owner.ResetReward();
            }

            private void AnimateObjects()
            {
                if (Owner.IsWorldFrozen)
                    return;

                foreach (GameObject item in Owner.GameObjects)
                {
                    IAnimated animatedItem = item as IAnimated;
                    if (animatedItem == null)
                        continue;

                    // Debug.Assert(animatedItem.AnimationEnumerator != null, "Animation enumerator is not initialized!");

                    //AnimationItem animation = animatedItem.AnimationEnumerator.Current;
                    AnimationItem animation = animatedItem.Current;

                    if (animation.Condition != null && !animation.Condition())
                    {
                        animatedItem.MoveNext();
                        animation = animatedItem.Current;
                    }

                    animation.StartAnimation(item, Owner);

                    if (!animation.IsStarted)
                        animation.StartAnimation(item, Owner);

                    switch (animation.Type)
                    {
                        case AnimationType.Translation:
                            {
                                Debug.Assert(animation.Data.Length >= 2, "Not enough data in animation data vector.");
                                item.Position.X += (int)animation.Data[0];
                                item.Position.Y += (int)animation.Data[1];
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
                for (int i = 0; i < Owner.GameObjects.Count; i++)
                {
                    if (Owner.GameObjects[i] is MovableGameObject)
                    {
                        GameObject obj = Owner.GameObjects[i];
                        MovableGameObject mobj = obj as MovableGameObject;

                        obj.Position.X += (int)(mobj.Velocity.X * Owner.Time);      // Apply horizontal velocity to X position
                        obj.Position.Y += (int)(mobj.Velocity.Y * Owner.Time);      // Apply vertical velocity to Y position
                    }
                }
            }

            public void HandleCollisions()
            {
                // detect collisions of objects that IsMoveableByPhysics() with any other objects.
                // When a collision is detected, handle it (do nothing, bounce, stop)

                foreach (GameObject obj in Owner.GameObjects)
                {
                    MovableGameObject mobj = obj as MovableGameObject;
                    if (mobj == null)
                        continue;

                    mobj.OnGround = false;
                }

                // the object queue length is a constant that represents how many objects are allowed to push on each other
                // before the engine stops repositioning them correctly
                for (int iObjectQueueCounter = 0; iObjectQueueCounter < 2; iObjectQueueCounter++)
                {
                    // Check if agent is colliding with any of the objects in Owner.GameObjects.
                    // If it is, adjust its postion to a position so that it doesn't collide
                    for (int i = 0; i < Owner.GameObjects.Count; i++)
                    {
                        GameObject obj = Owner.GameObjects[i];
                        MovableGameObject mobj = obj as MovableGameObject;
                        if (mobj == null)
                            continue;

                        mobj.ActualCollisions = new List<GameObject>();

                        for (int j = 0; j < Owner.GameObjects.Count; j++) // collisions with the remaining objects
                        {
                            if (i == j)
                                continue;
                            GameObject gameObj = Owner.GameObjects[j];
                            if (CheckCollision(mobj, gameObj))
                            {
                                Owner.m_conflictResolver.Resolve(mobj, gameObj);
                            }
                        }

                        // collisions with world boundaries
                        if (CheckCollision(mobj, m_floor)) Owner.m_conflictResolver.Resolve(mobj, m_floor);
                        if (CheckCollision(mobj, m_ceiling)) Owner.m_conflictResolver.Resolve(mobj, m_ceiling);
                        if (CheckCollision(mobj, m_rightSide)) Owner.m_conflictResolver.Resolve(mobj, m_rightSide);
                        if (CheckCollision(mobj, m_leftSide)) Owner.m_conflictResolver.Resolve(mobj, m_leftSide);
                        //MyLog.DEBUG.WriteLine("grounded: " + PlumberOwner.OnGround);
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

            public static PointF ReturnCoordinatesBetweenTwoPoints(PointF p1, PointF p2, float ratio)
            {
                return new PointF(p1.X + ratio * (p2.Y - p1.X), p1.Y + ratio * (p2.Y - p1.Y));
            }
        }

        #endregion

        #region Rendering

        /// <summary>
        /// Render the world to the visual output.
        /// </summary>
        public class RenderGLTask : MyTask<ManInWorld>
        {
            private MyCudaKernel m_addRgbNoiseKernel;

            INativeWindow m_window;
            IGraphicsContext m_context;

            uint m_fboHandle;
            uint m_renderTextureHandle;
            uint m_backgroundTexHandle;

            private uint m_sharedBufferHandle;
            private CudaOpenGLBufferInteropResource m_renderResource;

            private Dictionary<String, int> m_textureHandles;
            bool m_glInitialized;

            private static bool m_isToolkitInitialized;


            public override void Init(int nGPU)
            {
                m_addRgbNoiseKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "AddRgbNoiseKernel");

                m_textureHandles = new Dictionary<string, int>();
                m_glInitialized = false;

                // A hack to prevent BS from crashing after init
                Owner.VisualPOW.ExternalPointer =
                    MyMemoryManager.Instance.GetGlobalVariable("HACK_NAME_" + GetHashCode(), Owner.GPU, () => new float[Owner.VisualPOW.Count]).DevicePointer.Pointer;
            }

            public override void Execute()
            {
                if (!m_glInitialized)
                {
                    if (!m_isToolkitInitialized)
                    {
                        Toolkit.Init();
                        m_isToolkitInitialized = true;
                    }

                    // Clean the residual memory from init
                    MyMemoryManager.Instance.ClearGlobalVariable("HACK_NAME_" + GetHashCode(), Owner.GPU);
                    InitGL();
                    m_glInitialized = true;
                }

                m_context.MakeCurrent(m_window.WindowInfo);
                GL.Finish();

                // init textures
                UpdateTextures();

                SetupPoWview();

                RenderBackground();
                RenderGl();

                CopyPixelsPow();

                m_context.MakeCurrent(null);
            }

            void InitGL()
            {
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
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Owner.Pow.Width, Owner.Pow.Height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);

                // Setup background texture
                m_backgroundTexHandle = (uint)GL.GenTexture();
                GL.BindTexture(TextureTarget.Texture2D, m_backgroundTexHandle);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);

                string bitmapPath = "Ground_TOP.png";
                Owner.LoadAndGetBitmapSize(bitmapPath);
                Bitmap bmp = Owner.m_bitmapTable[bitmapPath].Item1;
                BitmapData data = bmp.LockBits(
                    new Rectangle(0, 0, bmp.Width, bmp.Height),
                    ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, data.Width, data.Height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, data.Scan0);
                bmp.UnlockBits(data);

                // Setup FBO
                m_fboHandle = (uint)GL.GenFramebuffer();
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, m_fboHandle);
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, m_renderTextureHandle, 0);

                // Setup Cuda <-> OpenGL interop
                int length = Owner.Pow.Width * Owner.Pow.Height * sizeof(uint);
                //unbind - just in case this is causing us the invalid exception problems
                GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                //create buffer
                GL.GenBuffers(1, out m_sharedBufferHandle);
                GL.BindBuffer(BufferTarget.PixelPackBuffer, m_sharedBufferHandle);
                GL.BufferData(BufferTarget.PixelPackBuffer, (IntPtr)length, IntPtr.Zero, BufferUsageHint.StaticRead);  // use data instead of IntPtr.Zero if needed
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

            void UpdateTextures()
            {
                foreach (var gameObject in Owner.GameObjects.Where(gameObject => !gameObject.IsBitmapAsMask && gameObject.BitmapPath != null))
                {
                    int loadedTextureHandle;

                    // We are assuming the gameObject.BitmapPath is the most up-to-date information about what should be rendered
                    if (!m_textureHandles.TryGetValue(gameObject.BitmapPath, out loadedTextureHandle))
                    {
                        // generate handle for new texture
                        GL.GenTextures(1, out loadedTextureHandle);
                        m_textureHandles.Add(gameObject.BitmapPath, loadedTextureHandle);

                        // load the Bitmap for the texture here
                        GL.BindTexture(TextureTarget.Texture2D, loadedTextureHandle);
                        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
                        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                        Owner.LoadAndGetBitmapSize(gameObject.BitmapPath);
                        Bitmap bmp = Owner.m_bitmapTable[gameObject.BitmapPath].Item1;
                        BitmapData data = bmp.LockBits(
                            new Rectangle(0, 0, gameObject.BitmapPixelSize.Width, gameObject.BitmapPixelSize.Height),
                            ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

                        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, data.Width, data.Height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, data.Scan0);

                        bmp.UnlockBits(data);
                        GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
                    }

                    // update texture for the gameObject
                    gameObject.SpriteTextureHandle = loadedTextureHandle;
                }

                GL.BindTexture(TextureTarget.Texture2D, 0);
            }

            void SetupPoWview()
            {
                PointF powCenter = Owner.GetPowCenter();

                // Setup view
                GL.Viewport(0, 0, Owner.Pow.Width, Owner.Pow.Height);

                GL.MatrixMode(MatrixMode.Projection);
                GL.LoadIdentity();
                SizeF viewportSize = new SizeF(Owner.Viewport.Width / 2, Owner.Viewport.Height / 2);
                GL.Ortho(powCenter.X - viewportSize.Width, powCenter.X + viewportSize.Width, powCenter.Y - viewportSize.Height, powCenter.Y + viewportSize.Height, -1, 1);
                GL.MatrixMode(MatrixMode.Modelview);
                GL.LoadIdentity();

                // Setup rendering
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, m_fboHandle);

                GL.Enable(EnableCap.Texture2D);
                GL.Enable(EnableCap.Blend);

                GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

                GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

                //GL.ClearColor(Owner.BackgroundColor);                

                GL.End();
            }

            void RenderBackground()
            {
                GL.PushMatrix();

                if (Owner.UseBackgroundTexture)
                {
                    GL.BindTexture(TextureTarget.Texture2D, m_backgroundTexHandle);
                    GL.Begin(PrimitiveType.Quads);
                    GL.TexCoord2(0.0f, 0.0f); GL.Vertex2(0f, 0f);
                    GL.TexCoord2(0.5f, 0.0f); GL.Vertex2(Owner.Scene.Width, 0f);
                    GL.TexCoord2(0.5f, 0.5f); GL.Vertex2(Owner.Scene.Width, Owner.Scene.Height);
                    GL.TexCoord2(0.0f, 0.5f); GL.Vertex2(0f, Owner.Scene.Height);
                    GL.BindTexture(TextureTarget.Texture2D, 0);
                }
                else
                {
                    GL.BindTexture(TextureTarget.Texture2D, m_backgroundTexHandle);
                    GL.ClearColor(Owner.BackgroundColor);
                    GL.Clear(ClearBufferMask.ColorBufferBit);
                    GL.BindTexture(TextureTarget.Texture2D, 0);
                }

                GL.End();

                GL.PopMatrix();
            }

            void RenderGl()
            {
                // Render game objects
                // TODO: object rendering order -- environment first, then creatures and active objects
                foreach (var gameObject in Owner.GameObjects)
                {
                    GL.PushMatrix();

                    // translate object to its position in the scene
                    GL.Translate(gameObject.Position.X, gameObject.Position.Y, 0.0f);
                    GL.Scale(gameObject.Size.Width, gameObject.Size.Height, 1f);

                    // translate back
                    GL.Translate(0.5f, 0.5f, 0.0f);
                    // rotate around center (origin)
                    GL.Rotate(gameObject.Rotation, 0.0f, 0.0f, 1.0f);
                    // translate s.t. object center in origin
                    GL.Translate(-0.5f, -0.5f, 0.0f);

                    Shape shape = gameObject as Shape;

                    if (shape != null && gameObject.IsBitmapAsMask)
                    {
                        // gameObject is a shape -> draw it directly
                        //((Shape)gameObject).ShapeType = Shape.Shapes.Triangle;
                        GL.BindTexture(TextureTarget.Texture2D, m_renderTextureHandle);
                        DrawShape(shape);
                    }
                    else if (gameObject.BitmapPath != null)
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

            void CopyPixelsPow()
            {
                // Prepare the results for CUDA
                // deinit CUDA interop to enable copying
                if (m_renderResource.IsMapped)
                    m_renderResource.UnMap();

                // bind pixel buffer object
                GL.BindBuffer(BufferTarget.PixelPackBuffer, m_sharedBufferHandle);
                // bind buffer from which data will be read
                GL.ReadBuffer(ReadBufferMode.ColorAttachment0);
                // read data to PBO (IntPtr.Zero means offset is 0)
                GL.ReadPixels(0, 0, Owner.Pow.Width, Owner.Pow.Height, OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedInt8888Reversed, IntPtr.Zero);
                GL.ReadBuffer(ReadBufferMode.None);

                GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);

                // Update the pointer for other usage in BS
                m_renderResource.Map();
                Owner.VisualPOW.ExternalPointer = m_renderResource.GetMappedPointer<uint>().DevicePointer.Pointer;
                Owner.VisualPOW.FreeDevice();
                Owner.VisualPOW.AllocateDevice();

                // add noise over POW
                if (Owner.IsImageNoise)
                {
                    MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal32(Owner.AgentVisualTemp.GetDevice(Owner).DevicePointer, Owner.AgentVisualTemp.Count, Owner.ImageNoiseMean, Owner.ImageNoiseStandardDeviation);

                    m_addRgbNoiseKernel.SetupExecution(Owner.Pow.Width * Owner.Pow.Height);
                    m_addRgbNoiseKernel.Run(Owner.VisualPOW, Owner.Pow.Width, Owner.Pow.Height, Owner.AgentVisualTemp);
                }
            }

            void DrawShape(Shape shape)
            {
                GL.Color4(shape.ColorMask);
                GL.Begin(PrimitiveType.Polygon);

                switch (shape.ShapeType)
                {
                    case Shape.Shapes.Circle:
                        DrawCircle();
                        break;
                    case Shape.Shapes.Square:
                        DrawSquare();
                        break;
                    case Shape.Shapes.Triangle:
                        DrawTriangle();
                        break;
                    case Shape.Shapes.Star:
                        DrawStar();
                        break;
                    case Shape.Shapes.Pentagon:
                        DrawPentagon();
                        break;
                    case Shape.Shapes.Mountains:
                        DrawMountains();
                        break;
                    case Shape.Shapes.T:
                        DrawT();
                        break;
                    case Shape.Shapes.Tent:
                        DrawTent();
                        break;
                    case Shape.Shapes.Rhombus:
                        DrawRhombus();
                        break;
                    case Shape.Shapes.DoubleRhombus:
                        DrawDoubleRhombus();
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

            void DrawTriangle()
            {
                GL.Vertex2(0, 0);
                GL.Vertex2(1, 0);
                GL.Vertex2(0.5f, 0.707f);  // 0.5, 1/sqrt(2)
            }

            void DrawSquare()
            {
                GL.Vertex2(0f, 0f);
                GL.Vertex2(1f, 0f);
                GL.Vertex2(1f, 1f);
                GL.Vertex2(0f, 1f);
            }

            void DrawCircle()
            {
                float deg2rad = 3.14159f / 180;
                for (int i = 0; i < 360; i++)
                {
                    float degInRad = i * deg2rad;
                    GL.Vertex2((Math.Cos(degInRad) + 1) / 2, (Math.Sin(degInRad) + 1) / 2);
                }
            }

            void DrawPentagon()
            {
                GL.Vertex2(1.0, 0.5);
                GL.Vertex2(0.654507120060765305, 0.97552870560096394);
                GL.Vertex2(0.095489800597887, 0.793890283234513885);
                GL.Vertex2(0.095489800597887, 0.206109716765486115);
                GL.Vertex2(0.654514005681348905, 0.024473531705853315);
            }

            void DrawStar()
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

            void DrawMountains()
            {
                GL.Vertex2(0.5f, 0.5f);
                GL.Vertex2(0.66f, 0f);
                GL.Vertex2(1f, 1f);
                GL.Vertex2(0f, 1);
                GL.Vertex2(0.33f, 0f);
            }

            void DrawT()
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

            void DrawTent()
            {
                GL.Vertex2(0.5f, 0f);
                GL.Vertex2(1f, 0.5f);
                GL.Vertex2(1f, 1f);
                GL.Vertex2(0.5f, 0.5f);
                GL.Vertex2(0f, 1f);
                GL.Vertex2(0f, 0.5f);
            }

            void DrawRhombus()
            {
                GL.Vertex2(0.33f, 0f);
                GL.Vertex2(1f, 0f);
                GL.Vertex2(0.66f, 1f);
                GL.Vertex2(0f, 1f);
            }

            void DrawDoubleRhombus()
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
                if (m_window == null)
                    return;

                try
                {
                    if (!m_context.IsDisposed && !m_context.IsCurrent && !m_window.Exists)
                        return;

                    m_context.MakeCurrent(m_window.WindowInfo);

                    ErrorCode err = GL.GetError();
                    if (err != ErrorCode.NoError)
                        MyLog.WARNING.WriteLine(Owner.Name + ": OpenGL error detected when disposing stuff, code: " + err);

                    // delete textures
                    if (m_textureHandles != null)
                    {
                        foreach (int handle in m_textureHandles.Values)
                        {
                            int h = handle;
                            GL.DeleteTextures(1, ref h);
                        }

                        m_textureHandles.Clear();
                    }

                    if (m_renderTextureHandle != 0)
                    {
                        GL.DeleteTextures(1, ref m_renderTextureHandle);
                        m_renderTextureHandle = 0;
                    }

                    // delete FBO
                    if (m_fboHandle != 0)
                    {
                        GL.DeleteFramebuffers(1, ref m_fboHandle);
                        m_fboHandle = 0;
                    }

                    // delete PBO
                    if (m_sharedBufferHandle != 0)
                    {
                        GL.DeleteBuffers(1, ref m_sharedBufferHandle);
                        m_sharedBufferHandle = 0;
                    }

                    // delete CUDA <-> GL interop
                    if (m_renderResource != null)
                    {
                        m_renderResource.Dispose();
                        m_renderResource = null;
                    }

                    if (m_context != null)
                    {
                        m_context.Dispose();
                        m_context = null;
                    }
                    if (m_window != null)
                    {
                        m_window.Dispose();
                        m_window = null;
                    }
                }
                catch (AccessViolationException e)
                {
                    MyLog.WARNING.WriteLine(Owner.Name + ": Failed when disposing OpenGL stuff. Cautious progress advised. Error: " + e.Message);
                }
                catch (Exception e)
                {
                    MyLog.WARNING.WriteLine(Owner.Name + ": Failed when disposing OpenGL. Error: " + e.Message);
                }
            }
        }

        #endregion

        #endregion
    }
}
