using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;

namespace Graph
{
    public class GraphStyle
    {
        public struct ArrowStyle
        {
            public float MinWidth;
            public bool FancyShortArrows;
            public bool WidenAtStart;
            public bool WidenAtEnd;
        }

        public ArrowStyle Arrow;

        public GraphStyle()
        {
            Arrow.MinWidth = 120;
            Arrow.FancyShortArrows = true;
            Arrow.WidenAtStart = false;
            Arrow.WidenAtEnd = true;
        }

        public PointF[] GetArrowPoints(float x, float y, float extra_thickness = 0)
        {
            return new PointF[]{
                    new PointF(x - (GraphConstants.ConnectorSize) - extra_thickness, y + (GraphConstants.ConnectorSize / 2.0f) + extra_thickness),
                    new PointF(x + 1.0f + extra_thickness, y),
                    new PointF(x - (GraphConstants.ConnectorSize) - extra_thickness, y - (GraphConstants.ConnectorSize / 2.0f) - extra_thickness)};
        }

        public List<PointF> GetArrowLinePoints(float x1, float y1, float x2, float y2, out float centerX, out float centerY, float extra_thickness = 0)
        {
            var widthX = (x2 - x1);
            var lengthX = Math.Max(Arrow.MinWidth / 2, Math.Abs(widthX / 2))
                //+ Math.Max(0, -widthX / 2)
                ;
            var lengthY = 0;// Math.Max(-170, Math.Min(-120.0f, widthX - 120.0f)) + 120.0f; 
            if (widthX < Arrow.MinWidth)
                lengthX = Arrow.MinWidth / 2;
            var yB = ((y1 + y2) / 2) + lengthY;// (y2 + ((y1 - y2) / 2) * 0.75f) + lengthY;
            var yC = y2 + yB;
            var xC = (x1 + x2) / 2;
            var xA = x1 + lengthX;
            var xB = x2 - lengthX;

            /*
            if (widthX >= 120)
            {
                xA = xB = xC = x2 - 60;
            }
            //-/



            /*
            if (widthX >= 120)
            {
                xA = xB = xC = x2 - 60;
            }
            //*/

            var points = new List<PointF> { 
                new PointF(x1, y1),
                new PointF(xA, y1),
                new PointF(xB, y2),
                new PointF(x2 - GraphConstants.ConnectorSize - extra_thickness, y2)
            };

            var t = 1.0f;//Math.Min(1, Math.Max(0, (widthX - 30) / 60.0f));
            var yA = (yB * t) + (yC * (1 - t));

            if (Arrow.FancyShortArrows && widthX <= 120)
            {
                points.Insert(2, new PointF(xB, yA));
                points.Insert(2, new PointF(xC, yA));
                points.Insert(2, new PointF(xA, yA));
            }
            
            using (var tempPath = new GraphicsPath())
            {
                tempPath.AddBeziers(points.ToArray());
                tempPath.Flatten();
                points = tempPath.PathPoints.ToList();
            }
            //
            var angles = new PointF[points.Count - 1];
            var lengths = new float[points.Count - 1];
            float totalLength = 0;
            centerX = 0;
            centerY = 0;
            points.Add(points[points.Count - 1]);
            for (int i = 0; i < points.Count - 2; i++)
            {
                var pt1 = points[i];
                var pt2 = points[i + 1];
                var pt3 = points[i + 2];
                var deltaX = (float)((pt2.X - pt1.X) + (pt3.X - pt2.X));
                var deltaY = (float)((pt2.Y - pt1.Y) + (pt3.Y - pt2.Y));
                var length = (float)Math.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
                if (length <= 1.0f)
                {
                    points.RemoveAt(i);
                    i--;
                    continue;
                }
                lengths[i] = length;
                totalLength += length;
                angles[i].X = deltaX / length;
                angles[i].Y = deltaY / length;
            }

            float midLength = (totalLength / 2.0f);// * 0.75f;
            float startWidth = extra_thickness + (Arrow.WidenAtStart ? (GraphConstants.ConnectorSize / 3.5f) : 0.75f);
            float endWidth = extra_thickness + (Arrow.WidenAtEnd ? (GraphConstants.ConnectorSize / 3.5f) : 0.75f);
            float currentLength = 0;
            var newPoints = new List<PointF>();
            newPoints.Add(points[0]);
            for (int i = 0; i < points.Count - 2; i++)
            {
                var angle = angles[i];
                var point = points[i + 1];
                var length = lengths[i];
                var width = (((currentLength * (endWidth - startWidth)) / totalLength) + startWidth);
                var angleX = angle.X * width;
                var angleY = angle.Y * width;

                var newLength = currentLength + length;
                if (currentLength <= midLength &&
                    newLength >= midLength)
                {
                    var dX = point.X - points[i].X;
                    var dY = point.Y - points[i].Y;
                    var t1 = midLength - currentLength;
                    var l = length;



                    centerX = points[i].X + ((dX * t1) / l);
                    centerY = points[i].Y + ((dY * t1) / l);
                }

                var pt1 = new PointF(point.X - angleY, point.Y + angleX);
                var pt2 = new PointF(point.X + angleY, point.Y - angleX);
                if (Math.Abs(newPoints[newPoints.Count - 1].X - pt1.X) > 1.0f ||
                    Math.Abs(newPoints[newPoints.Count - 1].Y - pt1.Y) > 1.0f)
                    newPoints.Add(pt1);
                if (Math.Abs(newPoints[0].X - pt2.X) > 1.0f ||
                    Math.Abs(newPoints[0].Y - pt2.Y) > 1.0f)
                    newPoints.Insert(0, pt2);

                currentLength = newLength;
            }

            return newPoints;
        }

    }
}
