using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
//using ManagedCuda.VectorTypes;
using System;
using System.Drawing;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Modules.School.Common
{
    public enum GameObjectType
    {
        None = 0,
        Agent = 1,
        Enemy = 2,
        Obstacle = 3,
        ClosedDoor = 4,
        OpenedDoor = 5,
        Teacher = 6,
        NonColliding = 7,
        OtherWithSubtype = 8
    };

    public enum GameObjectStyleType
    {
        None = 0,
        Platformer = 1,
        Pinball = 2,
    }

    public enum AnimationType
    {
        None = 0,
        Translation = 1,
    }

    // returns true if animation should continue and false if it should end and next animation should start
    public delegate bool ShouldContinue();

    public class AnimationItem
    {
        public AnimationType Type { get; private set; }
        public ShouldContinue Condition { get; private set; }
        public float[] Data { get; private set; }   // transformation-specific data (describe the transformation)

        public AnimationItem(AnimationType type = AnimationType.None, ShouldContinue condition = null, float[] data = null)
        {
            Type = type;
            Data = data;
            IsStarted = false;
        }

        public void StartAnimation(GameObject item, MyWorld world)
        {
            uint t = world.ExecutionBlock.SimulationStep;

            // default condition - run for 1 step only
            Condition = Condition ?? (() =>
            {
                if (world.ExecutionBlock.SimulationStep > t)
                    return false;
                return true;
            });

            IsStarted = true;
        }

        public bool IsStarted { get; private set; }
    }

    public interface IAnimated : IEnumerator<AnimationItem> { }

    public class GameObject
    {
        public int ID;
        public GameObjectType Type;
        public string Subtype;

        public Size BitmapPixelSize;
        public bool IsBitmapAsMask; // can use Bitmap's A value as mask
        protected Color m_colorMask;
        public Color ColorMask
        {
            get { return m_colorMask; }
            set { m_colorMask = value; }
        }

        public string BitmapPath;
        public CUdeviceptr BitmapPtr;
        public int SpriteTextureHandle;
        public int Layer = 0;

        public PointF Position;
        public SizeF Size;
        public float Rotation { get; set; } // rotation in radians

        public GameObject(string bitmapPath, PointF position = default(PointF), SizeF size = default(SizeF), GameObjectType type = GameObjectType.None, float rotation = 0, string subtype = null)
        {
            Type = type;
            BitmapPath = bitmapPath;
            Subtype = subtype;
            SpriteTextureHandle = -1;

            m_colorMask = Color.FromArgb(0, 0, 0);

            Position = position;
            Size = size;
            Rotation = rotation;
        }

        public float[] ToArray()
        {
            return new[] { ID, (int)Type, Position.X, Position.Y, Size.Width, Size.Height };
        }

        public int ArraySize
        {
            get
            {
                return 6;   // must be equal to length of ToArray representation in DWORDS
            }
        }

        public RectangleF GetGeometry()
        {
            return new RectangleF(new PointF(Position.X, Position.Y), new SizeF(Size.Width, Size.Height));
        }

        /// <summary>
        /// Computes the shortest distance from any bounding box pixel of this to any bounding box pixel of target
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public float DistanceTo(GameObject target)
        {
            // there are 4 configurations where corners are the closest pixels
            // there are also 4 configurations that have closest pixels on walls

            // right wall from left wall:
            if (Position.X + Size.Width < target.Position.X)
            {
                // bottom wall from top wall
                if (Position.Y + Size.Height < target.Position.Y)
                {
                    // distance of bottom right corner from top left corner:
                    return (float)Math.Sqrt(Math.Pow(Position.X + Size.Width - target.Position.X, 2) + Math.Pow(Position.Y + Size.Height - target.Position.Y, 2));
                }
                // top wall from bottom wall
                if (Position.Y > target.Position.Y + target.Size.Height)
                {
                    // distance of top right corner from bottom left corner:
                    return (float)Math.Sqrt(Math.Pow(Position.X + Size.Width - target.Position.X, 2) + Math.Pow(Position.Y - (target.Position.Y + target.Size.Height), 2));
                }
                // not corners, walls are closest together
                float rightLeftDist = target.Position.X - (Position.X + Size.Width);
                return rightLeftDist;
            }
            // left wall from right wall:
            if (Position.X > target.Position.X + target.Size.Width)
            {
                // bottom wall from top wall
                if (Position.Y + Size.Height < target.Position.Y)
                {
                    // distance of bottom left corner from top right corner:
                    return (float)Math.Sqrt(Math.Pow(Position.X - (target.Position.X + target.Size.Width), 2) + Math.Pow(Position.Y + Size.Height - target.Position.Y, 2));
                }
                // top wall from bottom wall
                if (Position.Y > target.Position.Y + target.Size.Height)
                {
                    // distance of top left corner from bottom right corner:
                    return (float)Math.Sqrt(Math.Pow(Position.X - (target.Position.X + target.Size.Width), 2) + Math.Pow(Position.Y - (target.Position.Y + target.Size.Height), 2));
                }
                // not corners, walls are closest together
                float leftRightDist = Position.X - (target.Position.X + target.Size.Width);
                return leftRightDist;
            }
            // the two game objects are above or below each other. Or intersecting.
            // bottom wall from top wall
            if (Position.Y + Size.Height < target.Position.Y)
            {
                float bottomTopDist = target.Position.Y - (Position.Y + Size.Height);
                return bottomTopDist;
            }
            // top wall from bottom wall
            if (Position.Y > target.Position.Y + target.Size.Height)
            {
                float topBottomDist = Position.Y - (target.Position.Y + target.Size.Width);
                return topBottomDist;
            }
            // the objects must intersect or touch:
            return 0;
        }

        public float CenterDistanceTo(GameObject target)
        {
            float centerX = Position.X + Size.Width / 2;
            float centerY = Position.Y + Size.Height / 2;
            float targetCenterX = target.Position.X + target.Size.Width / 2;
            float targetCenterY = target.Position.Y + target.Size.Height / 2;
            return (float)Math.Sqrt(Math.Pow(centerX - targetCenterX, 2) + Math.Pow(centerY - targetCenterY, 2));
        }

        public PointF GetCenter()
        {
            float centerX = Position.X + Size.Width / 2;
            float centerY = Position.Y + Size.Height / 2;
            return new PointF(centerX, centerY);
        }
    }

    public class MovableGameObject : GameObject
    {
        public PointF Velocity;
        public PointF VelocityPrevious;
        public PointF PositionPrevious;

        public bool OnGround { get; set; }      //OnGround tells whether a GameObject (that is IMovable) is standing on a surface or if it's in the air
        public GameObjectStyleType GameObjectStyle { get; set; }
        public bool IsAffectedByGravity { get; set; }
        public List<GameObject> ActualCollisions;


        public MovableGameObject(string bitmapPath, PointF position = default(PointF), SizeF size = default(SizeF), GameObjectType type = GameObjectType.None)
            : base(bitmapPath, position, size, type: type)
        {
            IsAffectedByGravity = true;
            GameObjectStyle = GameObjectStyleType.Platformer;
            ActualCollisions = new List<GameObject>();
        }

        public MovableGameObject(GameObjectType type, GameObjectStyleType style, string bitmapPath, PointF position = default(PointF), SizeF size = default(SizeF))
            : base(bitmapPath, position, size, type: type)
        {
            IsAffectedByGravity = true;
            GameObjectStyle = style;
            ActualCollisions = new List<GameObject>();
        }

        public bool IsMoving()
        {
            return Velocity.X != 0 || Velocity.Y != 0;
        }
    }

    public class AnimatedGameObject : GameObject, IAnimated
    {
        private List<AnimationItem> m_animation { get; set; }
        private int m_currentAnimationIndex;


        AnimationItem IEnumerator<AnimationItem>.Current
        {
            get
            {
                return m_animation[m_currentAnimationIndex];
            }
        }

        object IEnumerator.Current
        {
            get
            {
                return m_animation[m_currentAnimationIndex];
            }
        }

        public bool MoveNext()
        {
            m_currentAnimationIndex = (m_currentAnimationIndex + 1) % m_animation.Count;
            return true;
        }

        public void Reset()
        {
            m_currentAnimationIndex = 0;
        }

        public void Dispose() { }

        public AnimatedGameObject(GameObjectType type, string bitmapPath, PointF position, SizeF size = default(SizeF))
            : base(bitmapPath, position, size, type: type)
        {
            m_animation = new List<AnimationItem>();
            m_currentAnimationIndex = 0;
        }

        public void AddAnimationItem(AnimationItem item)
        {
            m_animation.Add(item);
        }
    }

    public interface ISwitchable
    {
        bool IsOn { get; set; }

        void Switch();
        void Switch(bool on);
        bool SwitchOnCollision();
    }
}
