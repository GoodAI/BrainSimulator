using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class ScFixPositions
    {
        public List<PointF> Positions;

        public ScFixPositions(RectangleF powGeometry)
        {
            Positions = new List<PointF>
            {
                new PointF(0.0f,0.0f),
                new PointF(0.0f,1.0f),
                new PointF(0.0f,2.0f),
                new PointF(1.0f,0.0f),
                new PointF(1.0f,2.0f),
                new PointF(2.0f,0.0f),
                new PointF(2.0f,1.0f),
                new PointF(2.0f,2.0f)
            };

            for (int i = 0; i < Positions.Count; i++)
            {
                float marginX = powGeometry.Width / 16;
                float marginY = powGeometry.Height / 16;
                PointF position = Positions[i];
                position.X = marginX + position.X * ((powGeometry.Width - marginX) / 3) + powGeometry.X;
                position.Y = marginY + position.Y * ((powGeometry.Height - marginY) / 3) + powGeometry.Y;
                Positions[i] = position;
            }
        }

        public PointF GetRandomPosition(Random rnd)
        {
            return Positions[rnd.Next(Positions.Count)];
        }
    }
}