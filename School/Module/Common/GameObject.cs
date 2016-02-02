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
        public AnimationType type { get; private set; }
        public ShouldContinue condition { get; private set; }
        public float[] data { get; private set; }   // transformation-specific data (describe the transformation)

        public AnimationItem(AnimationType type = AnimationType.None, ShouldContinue condition = null, float[] data = null)
        {
            this.type = type;
            this.data = data;
            IsStarted = false;
        }

        public void StartAnimation(GameObject item, MyWorld world)
        {
            uint t = world.ExecutionBlock.SimulationStep;

            // default condition - run for 1 step only
            this.condition = condition ?? (() =>
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
        public int id;
        public GameObjectType type;
        public string Subtype;
        public Size bitmapPixelSize;
        public bool isBitmapAsMask; // can use bitmap's A value as mask
        public Color maskColor; // R,G,B 
        public string bitmapPath;
        public CUdeviceptr bitmap;
        public int SpriteTextureHandle;

        public int X { get; set; }
        public int Y { get; set; }

        public float Rotation { get; set; } // rotation in radians

        public int Width { get; set; }
        public int Height { get; set; }

        public GameObject(GameObjectType type, string path, int x, int y, int width = 0, int height = 0, string subtype = null)
        {
            this.type = type;
            this.bitmapPath = path;
            this.X = x;
            this.Y = y;
            this.Rotation = 0;
            this.bitmapPixelSize = new Size(0, 0);
            this.Width = width;
            this.Height = height;
            this.Subtype = subtype;

            isBitmapAsMask = false;
            maskColor = Color.FromArgb(0, 0, 0);
        }

        public int[] ToArray()
        {
            return new int[] { id, (int)type, X, Y, Width, Height };
        }

        public int ArraySize
        {
            get
            {
                return 6;   // must be equal to length of ToArray representation in DWORDS
            }
        }

        public Rectangle GetGeometry()
        {
            return new Rectangle(new Point(X,Y), new Size(Width,Height));
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
            if (X + Width < target.X)
            {
                // bottom wall from top wall
                if (Y + Height < target.Y)
                {
                    // distance of bottom right corner from top left corner:
                    return (float)Math.Sqrt(Math.Pow(X + Width - target.X, 2) + Math.Pow(Y + Height - target.Y, 2));
                }
                // top wall from bottom wall
                if (Y > target.Y + target.Height)
                {
                    // distance of top right corner from bottom left corner:
                    return (float)Math.Sqrt(Math.Pow(X + Width - target.X, 2) + Math.Pow(Y - (target.Y + target.Height), 2));
                }
                // not corners, walls are closest together
                int rightLeftDist = target.X - (X + Width);
                return rightLeftDist;
            }
            // left wall from right wall:
            if (X > target.X + target.Width)
            {
                // bottom wall from top wall
                if (Y + Height < target.Y)
                {
                    // distance of bottom left corner from top right corner:
                    return (float)Math.Sqrt(Math.Pow(X - (target.X + target.Width), 2) + Math.Pow(Y + Height - target.Y, 2));
                }
                // top wall from bottom wall
                if (Y > target.Y + target.Height)
                {
                    // distance of top left corner from bottom right corner:
                    return (float)Math.Sqrt(Math.Pow(X - (target.X + target.Width), 2) + Math.Pow(Y - (target.Y + target.Height), 2));
                }
                // not corners, walls are closest together
                int leftRightDist = X - (target.X + target.Width);
                return leftRightDist;
            }
            // the two game objects are above or below each other. Or intersecting.
            // bottom wall from top wall
            if (Y + Height < target.Y)
            {
                int bottomTopDist = target.Y - (Y + Height);
                return bottomTopDist;
            }
            // top wall from bottom wall
            if (Y > target.Y + target.Height)
            {
                int topBottomDist = Y - (target.Y + target.Width);
                return topBottomDist;
            }
            // the objects must intersect or touch:
            return 0;
        }

        public float CenterDistanceTo(GameObject target)
        {
            int centerX = X + this.Width / 2;
            int centerY = Y + this.Height / 2;
            int targetCenterX = target.X + target.Width / 2;
            int targetCenterY = target.Y + target.Height / 2;
            return (float)Math.Sqrt(Math.Pow(centerX - targetCenterX, 2) + Math.Pow(centerY - targetCenterY, 2));
        }

        public void SetColor(int r, int g, int b)
        {
            this.maskColor = Color.FromArgb(r, g, b);
            this.isBitmapAsMask = true;
        }

        public void SetColor(Color c)
        {
            this.maskColor = c;
            this.isBitmapAsMask = true;
        }

        public void SetPosition(Point p)
        {
            X = p.X;
            Y = p.Y;
        }
    }

    public class MovableGameObject : GameObject
    {
        public float vX { get; set; }
        public float vY { get; set; }
        public int previousX { get; set; }      //previousX denotes the X position in the previous step of the simulation
        public int previousY { get; set; }      //previousY denotes the Y position in the previous step of the simulation
        public float previousvX { get; set; }   //previousX denotes the X velocity in the previous step of the simulation
        public float previousvY { get; set; }   //previousY denotes the Y velocity in the previous step of the simulation
        public bool onGround { get; set; }      //onGround tells whether a GameObject (that is IMovable) is standing on a surface or if it's in the air
        public GameObjectStyleType GameObjectStyle { get; set; }
        public bool IsAffectedByGravity { get; set; }
        public List<GameObject> ActualCollisions;

        public MovableGameObject(GameObjectType type, string path, int x, int y, int width = 0, int height = 0)
            : base(type, path, x, y, width, height)
        {
            vX = vY = 0;
            previousvX = previousvY = 0;
            previousX = previousY = 0;
            IsAffectedByGravity = true;
            GameObjectStyle = GameObjectStyleType.Platformer;
            ActualCollisions = new List<GameObject>();
        }

        public MovableGameObject(GameObjectType type, GameObjectStyleType style, string path, int x, int y, int width = 0, int height = 0)
            : base(type, path, x, y, width, height)
        {
            vX = vY = 0;
            previousvX = previousvY = 0;
            previousX = previousY = 0;
            IsAffectedByGravity = true;
            GameObjectStyle = style;
            ActualCollisions = new List<GameObject>();
        }

        public bool isMoving()
        {
            return vX != 0 || vY != 0;
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

        public AnimatedGameObject(GameObjectType type, string path, int x, int y, int width = 0, int height = 0)
            : base(type, path, x, y, width, height)
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
        bool isOn { get; set; }

        void SwitchOn();
        void SwitchOff();

        void Switch();
        bool SwitchOnCollision();
    }
}
