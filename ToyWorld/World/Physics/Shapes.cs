using System;
using System.Collections.Generic;
using VRageMath;

namespace World.Physics
{
    public interface IShape
    {
        Vector2 Position { get; set; }

        Vector2 Size { get; set; }

        /// <summary>
        /// Returns size of rectangle which can wrap this shape.
        /// </summary>
        /// <returns></returns>
        VRageMath.Vector2 CoverRectangleSize();

        /// <summary>
        /// Returns maximum distance from center to the furthermost point of this shape.
        /// </summary>
        /// <returns></returns>
        float PossibleCollisionDistance();

        /// <summary>
        /// Returns list of squares covered by this shape on given position.
        /// </summary>
        /// <returns></returns>
        List<Vector2I> CoverTiles();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="eps">Border width.</param>
        /// <returns></returns>
        bool CollidesWith(IShape shape);

        void Resize(float add);
    }

    public abstract class Shape : IShape
    {
        public Vector2 Position { get; set; }

        public Vector2 Size { get; set; }

        public abstract float PossibleCollisionDistance();
        
        public abstract Vector2 CoverRectangleSize();

        public abstract List<Vector2I> CoverTiles();

        public abstract bool CollidesWith(IShape shape);

        public abstract void Resize(float add);
    }

    public class RectangleShape : Shape
    {
        public RectangleShape(Vector2 size)
        {
            Size = size;
        }

        public override float PossibleCollisionDistance()
        {
            return Size.Length() / 2;
        }

        public override Vector2 CoverRectangleSize()
        {
            return Size;
        }

        public override List<Vector2I> CoverTiles()
        {
            Vector2 cornerUl = Position - Size/2;
            Vector2 cornerLr = cornerUl + Size;

            var list = new List<Vector2I>();

            for (int i = (int)Math.Floor(cornerUl.X); i < (int)Math.Ceiling(cornerLr.X); i++)
            {
                for (int j = (int)Math.Floor(cornerUl.Y); j < (int)Math.Ceiling(cornerLr.Y); j++)
                {
                    list.Add(new Vector2I(i, j));
                }
            }

            return list;
        }

        public override bool CollidesWith(IShape shape)
        {
            RectangleShape rectangle = shape as RectangleShape;
            if (rectangle != null)
            {
                RectangleF result;
                RectangleF thisRectangleF = VRectangle;
                RectangleF thatRectangleF = rectangle.VRectangle;
                RectangleF.Intersect(ref thisRectangleF, ref thatRectangleF, out result);
                return result.Size.X > 0 || result.Y > 0;
            }

            CircleShape circle = shape as CircleShape;
            if (circle != null)
            {
                return circle.CollidesWith(this);
            }

            throw new NotImplementedException();
        }

        public override void Resize(float add)
        {
            Size += add;
        }

        public RectangleF VRectangle
        {
            get { return new RectangleF(Position, Size); }
        }
    }

    public class CircleShape : Shape
    {
        public VRageMath.Circle VCircle { get; private set; }

        public float Radius { get; private set; }

        public CircleShape(float radius)
        {
            Radius = radius;
        }

        public CircleShape(Vector2 size)
        {
            Radius = (size.X + size.Y) / 4;
        }

        public override float PossibleCollisionDistance()
        {
            return Radius;
        }

        public override Vector2 CoverRectangleSize()
        {
            float side = Radius * 2;
            return new Vector2(side, side);
        }

        public override List<Vector2I> CoverTiles()
        {
            Vector2 coverRectangleSize = CoverRectangleSize();

            Vector2 cornerUL = Position - coverRectangleSize / 2;
            Vector2 cornerLR = cornerUL + coverRectangleSize;

            List<Vector2I> list = new List<Vector2I>();

            for (int i = (int)Math.Floor(cornerUL.X); i < (int)Math.Ceiling(cornerLR.X); i++)
            {
                for (int j = (int)Math.Floor(cornerUL.Y); j < (int)Math.Ceiling(cornerLR.Y); j++)
                {
                    list.Add(new Vector2I(i, j));
                }
            }

            list.RemoveAll(x => !CircleRectangleIntersects(new RectangleF(new Vector2(x.X, x.Y), Vector2.One)));

            return list;
        }

        public override bool CollidesWith(IShape shape)
        {
            CircleShape circle = shape as CircleShape;
            if (circle != null)
            {
                return Vector2.Distance(circle.Position, Position) < Radius + circle.Radius;
            }

            RectangleShape rectangle = shape as RectangleShape;
            if (rectangle != null)
            {
                return CircleRectangleIntersects(rectangle.VRectangle);
            }
            throw new NotImplementedException();
        }

        public override void Resize(float add)
        {
            Radius += add;
        }

        private bool CircleRectangleIntersects(RectangleF rectangle)
        {
            float rectangleCX = rectangle.X + rectangle.Width / 2;
            float rectangleCY = rectangle.Y + rectangle.Height / 2;
            float circleDistanceX = Math.Abs(Position.X - rectangleCX);
            float circleDistanceY = Math.Abs(Position.Y - rectangleCY);

            if (circleDistanceX > (rectangle.Width / 2 + Radius)) { return false; }
            if (circleDistanceY > (rectangle.Height / 2 + Radius)) { return false; }

            if (circleDistanceX <= (rectangle.Width / 2)) { return true; }
            if (circleDistanceY <= (rectangle.Height / 2)) { return true; }

            float cornerDistanceSq = (float) Math.Pow(circleDistanceX - rectangle.Width / 2, 2) +
                                   (float) Math.Pow(circleDistanceY - rectangle.Height / 2, 2);

            bool containsCorner = cornerDistanceSq <= Math.Pow(Radius, 2);

            return containsCorner;
        }
    }

    /// <summary>
    /// Given by x^2/a^2 + y^2/b^2 = 1
    /// </summary>
    public class EllipseShape : Shape
    {

        public override float PossibleCollisionDistance()
        {
            throw new NotImplementedException();
        }

        public override Vector2 CoverRectangleSize()
        {
            throw new NotImplementedException();
        }

        public override List<Vector2I> CoverTiles()
        {
            throw new NotImplementedException();
        }

        public override bool CollidesWith(IShape shape)
        {
            throw new NotImplementedException();
        }

        public override void Resize(float add)
        {
            throw new NotImplementedException();
        }
    }
}
