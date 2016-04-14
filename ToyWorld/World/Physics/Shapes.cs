using System;
using System.Collections.Generic;
using VRageMath;

namespace World.Physics
{
    public interface IShape
    {
        /// <summary>
        /// Given upper left corner, method returns center of shape.
        /// </summary>
        /// <param name="upperLefCorner"></param>
        /// <returns>Vector2 of center</returns>
        VRageMath.Vector2 Center(VRageMath.Vector2 upperLefCorner);

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
        List<Vector2I> CoverTiles(Vector2 position);
    }

    public abstract class Shape : IShape
    {
        abstract public float PossibleCollisionDistance();

        abstract public Vector2 Center(Vector2 upperLefCorner);

        abstract public Vector2 CoverRectangleSize();

        abstract public List<Vector2I> CoverTiles(Vector2 position);
    }

    public class Rectangle : Shape
    {
        public Vector2 Size { get; private set; }

        public Rectangle(Vector2 size)
        {
            Size = size;
        }

        public override float PossibleCollisionDistance()
        {
            return Size.Length() / 2;
        }

        public override Vector2 Center(Vector2 upperLefCorner)
        {
            return upperLefCorner + Size / 2;
        }

        public override Vector2 CoverRectangleSize()
        {
            return Size;
        }

        public override List<Vector2I> CoverTiles(Vector2 position)
        {
            Vector2 cornerUL = position;
            Vector2 cornerLR = position + Size;

            var list = new List<Vector2I>();

            for (int i = (int)Math.Floor(cornerUL.X); i < (int)Math.Ceiling(cornerLR.X); i++)
            {
                for (int j = (int)Math.Floor(cornerUL.Y); j < (int)Math.Ceiling(cornerLR.Y); j++)
                {
                    list.Add(new Vector2I(i, j));
                }
            }

            return list;
        }
    }

    public class Circle : Shape
    {
        public float Radius { get; private set; }

        public Circle(float radius)
        {
            Radius = radius;
        }

        public override float PossibleCollisionDistance()
        {
            return Radius;
        }

        public override Vector2 Center(Vector2 upperLefCorner)
        {
            return upperLefCorner + Radius;
        }

        public override Vector2 CoverRectangleSize()
        {
            float side = Radius * 2;
            return new Vector2(side, side);
        }

        public override List<Vector2I> CoverTiles(Vector2 position)
        {
            Vector2 center = Center(position);

            Vector2 coverRectangleSize = CoverRectangleSize();

            Vector2 cornerUL = position;
            Vector2 cornerLR = position + coverRectangleSize;

            List<Vector2I> list = new List<Vector2I>();

            for (int i = (int)Math.Floor(cornerUL.X); i < (int)Math.Ceiling(cornerLR.X); i++)
            {
                for (int j = (int)Math.Floor(cornerUL.Y); j < (int)Math.Ceiling(cornerLR.Y); j++)
                {
                    list.Add(new Vector2I(i, j));
                }
            }

            list.RemoveAll(x => !CircleRectangleIntersects(center, new RectangleF(x, Vector2.One)));

            return list;
        }

        private bool CircleRectangleIntersects(Vector2 center, RectangleF rectangle)
        {
            float rectangleCX = rectangle.X + rectangle.Width / 2;
            float rectangleCY = rectangle.Y + rectangle.Height / 2;
            float circleDistanceX = Math.Abs(center.X - rectangleCX);
            float circleDistanceY = Math.Abs(center.Y - rectangleCY);

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
    public class Ellipse : Shape
    {
        public float A { get; set; }
        public float B { get; set; }

        public override float PossibleCollisionDistance()
        {
            throw new NotImplementedException();
        }

        public override Vector2 Center(Vector2 upperLefCorner)
        {
            throw new NotImplementedException();
        }

        public override Vector2 CoverRectangleSize()
        {
            throw new NotImplementedException();
        }

        public override List<Vector2I> CoverTiles(Vector2 position)
        {
            throw new NotImplementedException();
        }
    }
}
