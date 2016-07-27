using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class ScFixPositions
    {
        public List<float[]> Positions;

        public ScFixPositions(RectangleF powGeometry)
        {
            Positions = new List<float[]>
            {
                new[] {0.0f, 0.0f},
                new[] {0.0f, 1.0f},
                new[] {0.0f, 2.0f},
                new[] {1.0f, 0.0f},
                new[] {1.0f, 2.0f},
                new[] {2.0f, 0.0f},
                new[] {2.0f, 1.0f},
                new[] {2.0f, 2.0f}
            };

            foreach (float[] position in Positions)
            {
                float marginX = powGeometry.Width / 16;
                float marginY = powGeometry.Height / 16;
                position[0] = marginX + position[0] * ((powGeometry.Width - marginX) / 3) + powGeometry.X;
                position[1] = marginY + position[1] * ((powGeometry.Height - marginY) / 3) + powGeometry.Y;
            }
        }

        public PointF GetRandomPosition(Random rnd)
        {
            int r = rnd.Next(Positions.Count);
            var p = Positions[r];
            return new PointF(p[0],p[1]);
        }
    }
}