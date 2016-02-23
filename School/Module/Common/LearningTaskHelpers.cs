using System;
using System.Drawing;
using System.Diagnostics;
using System.Collections.Generic;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System.Linq;

namespace GoodAI.Modules.School.Common
{
    public class Shape : GameObject
    {
        public enum Shapes { Circle, Square, Star, Triangle, Mountains, T, Tent, Pentagon, DoubleRhombus, Rhombus }

        public Shapes ShapeType { get; set; }

        public Shape(Shapes shapeType, PointF position, SizeF size = default(SizeF), float rotation = 0f, GameObjectType type = GameObjectType.None)
            : base(GetShapeAddr(shapeType), position, size, type: type, rotation: rotation)
        {
            ShapeType = shapeType;
        }

        public static string GetShapeAddr(Shapes shape)
        {
            switch (shape)
            {
                case Shapes.Circle:
                    return @"WhiteCircle50x50.png";
                case Shapes.Square:
                    return @"White10x10.png";
                case Shapes.Triangle:
                    return @"WhiteTriangle50x50.png";
                case Shapes.Star:
                    return @"WhiteStar50x50.png";
                case Shapes.Mountains:
                    return @"WhiteMountains50x50.png";
                case Shapes.T:
                    return @"WhiteT50x50.png";
                case Shapes.Tent:
                    return @"WhiteTent50x50.png";
                case Shapes.Pentagon:
                    return @"WhitePentagon50x50.png";
                case Shapes.DoubleRhombus:
                    return @"WhiteDoubleRhombus50x50.png";
                case Shapes.Rhombus:
                    return @"WhiteRhombus50x50.png";
            }
            throw new ArgumentException("Unknown shape");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rndGen"></param>
        /// <param name="numberOfShapes">Cardinality of set from you are choosing</param>
        /// <returns>Random shape</returns>
        public static Shapes GetRandomShape(Random rndGen, int numberOfShapes = 10)
        {
            Array values = Enum.GetValues(typeof(Shapes));
            if (numberOfShapes > values.Length)
            {
                throw new ArgumentException("Not Enought Shapes.");
            }
            return (Shapes)values.GetValue(rndGen.Next(numberOfShapes));
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

        public AbstractTeacherInWorld(string bitmapPath, PointF position = default(PointF), SizeF size = default(SizeF), GameObjectType type = GameObjectType.None)
            : base(bitmapPath, position, size, type)
        { }

    }

    public class Grid
    {
        private SizeF m_blockSize;
        private SizeF m_size;

        public Grid(SizeF size, SizeF blockSize)
        {
            m_size = size;
            m_blockSize = blockSize;
        }

        public PointF GetPoint(int xGrid, int yGrid)
        {
            float px = xGrid * m_blockSize.Width;
            float py = yGrid * m_blockSize.Height;
            if (px > m_size.Width || py > m_size.Height)
            {
                throw new ArgumentException("Out of Grid");
            }
            return new PointF(px, py);
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

        public static float GetRandomGaussian(Random rndGen)
        {
            float u1 = Convert.ToSingle(rndGen.NextDouble()); //these are uniform(0,1) random doubles
            float u2 = Convert.ToSingle(rndGen.NextDouble()); //these are uniform(0,1) random doubles
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2)); //random normal(0,1)
            return randStdNormal;
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

        public static Color RandomizeColorWDiff(Color backgroundColor, float minDifference, Random rndGen)
        {
            Debug.Assert(minDifference >= 0.0f && minDifference < 0.5f);
            byte dif = (byte)(256 * minDifference);

            int newR = rndGen.Next(256);
            if (Math.Abs(newR - backgroundColor.R) < dif)
            {
                newR = backgroundColor.R + dif;
                if (newR > 255)
                    newR -= 255;
            }
            int newG = rndGen.Next(256);
            if (Math.Abs(newG - backgroundColor.G) < dif)
            {
                newG = backgroundColor.G + dif;
                if (newG > 255)
                    newG -= 255;
            }
            int newB = rndGen.Next(256);
            if (Math.Abs(newB - backgroundColor.B) < dif)
            {
                newB = backgroundColor.B + dif;
                if (newB > 255)
                    newB -= 255;
            }
            return Color.FromArgb(newR, newG, newB);

        }

        public static Color RandomVisibleColor(Random rndGen, int numberOfColors = 11)
        {
            return GetVisibleColor(rndGen.Next(numberOfColors));
        }

        public static Color GetVisibleColor(int colorIndex)
        {
            switch (colorIndex)
            {
                case 0:
                    return Color.Red;
                case 1:
                    return Color.White;
                case 2:
                    return Color.Yellow;
                case 3:
                    return Color.Gray;
                case 4:
                    return Color.Black;
                case 5:
                    return Color.SteelBlue;
                case 6:
                    return Color.Green;
                case 7:
                    return Color.Orange;
                case 8:
                    return Color.Pink;
                case 9:
                    return Color.Brown;
                case 10:
                    return Color.Gold;
                default:
                    return Color.White;
            }
        }

        public static List<int> UniqueNumbers(Random rndGen, int lowerBound, int upperBound, int count)
        {
            if (upperBound - lowerBound < count)
            {
                throw new ArgumentException();
            }

            // generate count random values.
            HashSet<int> candidates = new HashSet<int>();
            for (Int32 top = upperBound - count; top < upperBound; top++)
            {
                // May strike a duplicate.
                if (!candidates.Add(lowerBound + rndGen.Next(top + 1 - lowerBound)))
                {
                    candidates.Add(top);
                }
            }

            // load them in to a list.
            List<int> result = candidates.ToList();

            // shuffle the results:
            int i = result.Count;
            while (i > 1)
            {
                i--;
                int k = rndGen.Next(i + 1);
                int value = result[k];
                result[k] = result[i];
                result[i] = value;
            }
            return result;
        }

        public static RectangleF ResizeRectangleAroundCentre(RectangleF r, float ratio)
        {
            RectangleF f = new RectangleF();

            float dX = (r.Width * ratio - r.Width) / 2;
            float dY = (r.Height * ratio - r.Height) / 2;

            float xNew = r.X - dX;
            float yNew = r.Y - dY;
            f.Location = new PointF(xNew, yNew);

            float newWidth = r.Width + 2 * dX;
            float newHeight = r.Height + 2 * dX;
            f.Size = new SizeF(newWidth, newHeight);

            return f;
        }

        public static RectangleF ResizeRectangleAroundCentre(RectangleF r, float dx, float dy)
        {
            RectangleF f = new RectangleF();

            float dX = dx;
            float dY = dy;

            float xNew = r.X - dX;
            float yNew = r.Y - dY;
            f.Location = new PointF(xNew, yNew);

            float newWidth = r.Width + 2 * dX;
            float newHeight = r.Height + 2 * dX;
            f.Size = new SizeF(newWidth, newHeight);

            return f;
        }
    }
}
