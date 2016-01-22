using System;
using System.Drawing;
using System.Diagnostics;
using System.Collections.Generic;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;

namespace GoodAI.Modules.School.Common
{
    public class Shape : GameObject
    {
        public enum Shapes { Circle, Square, Star, Triangle }

        public Shapes ShapeType { get; set; }

        public Shape(Shapes shapeType, int x, int y, int width = 0, int height = 0)
            : base(GameObjectType.None, GetShapeAddr(shapeType), x, y, width, height)
        {
            this.ShapeType = shapeType;
        }

        public Shape(Shapes shapeType, int x, int y, GameObjectType type, int width = 0, int height = 0)
            : base(type, GetShapeAddr(shapeType), x, y, width, height)
        {
            this.ShapeType = shapeType;
        }

        public static string GetShapeAddr(Shape.Shapes shape)
        {
            switch (shape)
            {
                case Shape.Shapes.Circle:
                    return @"WhiteCircle50x50.png";
                case Shape.Shapes.Square:
                    return @"White10x10.png";
                case Shape.Shapes.Triangle:
                    return @"WhiteTriangle50x50.png";
                case Shape.Shapes.Star:
                    return @"WhiteStar50x50.png";
            }
            throw new ArgumentException("Unknown shape");
        }
    }

    public abstract class AbstractTeacherInWorld : MovableGameObject
    {

        protected int m_currentMove;

        public abstract void Stop();
        public abstract void Reset();
        public abstract bool IsDone();
        public abstract int ActionsCount();
        public abstract float[] CurrentAction();

        public AbstractTeacherInWorld(GameObjectType type, string path, int x, int y, int width = 0, int height = 0)
            : base(type, path, x, y, width, height) { }

    }

    public class Grid
    {
        private Size m_blockSize;
        private Size m_size;

        public Grid(Size size, Size blockSize)
        {
            m_size = size;
            m_blockSize = blockSize;
        }

        public Point getPoint(int xGrid, int yGrid){
            int px = xGrid * m_blockSize.Width;
            int py = yGrid * m_blockSize.Height;
            if (px > m_size.Width || py > m_size.Height)
            {
                throw new ArgumentException("Out of Grid");
            }
            return new Point(px, py);
        }
    }

    public enum ColorType { Bluish, Yellowish };

    public static class LearningTaskHelpers
    {
        // Return true or false with equal probability
        public static bool FlipCoin(Random rndGen)
        {
            return rndGen.Next(0, 2) == 1;
        }

        public static void RandomizeColor(ref Color color, Random rndGen)
        {
            // Colors are uniformly distributed in either of two corners in
            // the RGB cube.
            ColorType colorType = FlipCoin(rndGen) ? ColorType.Bluish : ColorType.Yellowish;
            const byte SIZE = 255 / 2; // (size = 0.5f)

            switch (colorType)
            {
                case ColorType.Bluish:
                    color = Color.FromArgb(rndGen.Next(SIZE), rndGen.Next(SIZE), 255 - rndGen.Next(SIZE));
                    break;
                case ColorType.Yellowish:
                default:
                    color = Color.FromArgb(255 - rndGen.Next(SIZE), 255 - rndGen.Next(SIZE), rndGen.Next(SIZE));
                    break;
            }
        }

        public static void RandomizeColorWDiff(ref Color color, float minDifference, Random rndGen)
        {
            Debug.Assert(minDifference >= 0.0f && minDifference < 0.5f);
            byte dif = (byte)(256 * minDifference);

            int newR = rndGen.Next(256);
            if (Math.Abs(newR - color.R) < dif)
            {
                newR = color.R + dif;
                if (newR > 255)
                    newR -= 255;
            }
            int newG = rndGen.Next(256);
            if (Math.Abs(newG - color.G) < dif)
            {
                newG = color.G + dif;
                if (newG > 255)
                    newG -= 255;
            }
            int newB = rndGen.Next(256);
            if (Math.Abs(newB - color.B) < dif)
            {
                newB = color.B + dif;
                if (newB > 255)
                    newB -= 255;
            }
            color = Color.FromArgb(newR, newG, newB);
        }
    }
}
