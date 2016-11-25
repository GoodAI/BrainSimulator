using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    public class ScFixPositions
    {
        public List<PointF> Positions;
        public List<PointF> PositionsWithoutCenter;
        private RectangleF m_powGeometry;
        private Random m_rndGen;

        public ScFixPositions(RectangleF powGeometry)
        {
            m_powGeometry = powGeometry;
            m_rndGen = new Random();

            Positions = new List<PointF>
            {
                new PointF(0.0f,0.0f),
                new PointF(0.0f,1.0f),
                new PointF(0.0f,2.0f),
                new PointF(1.0f,0.0f),
                new PointF(1.0f,1.0f),
                new PointF(1.0f,2.0f),
                new PointF(2.0f,0.0f),
                new PointF(2.0f,1.0f),
                new PointF(2.0f,2.0f)
            };

            Debug.Assert(Positions.Count == ScConstants.numPositions);

            for (int i = 0; i < Positions.Count; i++)
            {
                float marginX = powGeometry.Width / 16;
                float marginY = powGeometry.Height / 16;
                PointF position = Positions[i];
                position.X = marginX + position.X * ((powGeometry.Width - marginX) / 3) + powGeometry.X;
                position.Y = marginY + position.Y * ((powGeometry.Height - marginY) / 3) + powGeometry.Y;
                Positions[i] = position;
            }

            PositionsWithoutCenter = Positions.GetRange(0, Positions.Count);
            PositionsWithoutCenter.RemoveAt(4);
        }

        public PointF GetRandomFreePosition()
        {
            float marginX = m_powGeometry.Width / 16;
            float marginY = m_powGeometry.Height / 16;
            PointF res = new PointF();
            res.X = marginX + (float)m_rndGen.NextDouble() * (m_powGeometry.Width - marginX) + m_powGeometry.X;
            res.Y = marginY + (float)m_rndGen.NextDouble() * (m_powGeometry.Height - marginY) + m_powGeometry.Y;
            return res;
        }

        public PointF GetRandomPosition(Random rnd)
        {
            return Positions[rnd.Next(Positions.Count)];
        }

        public PointF GetRandomPositionWithoutCenter(Random rnd)
        {
            return PositionsWithoutCenter[rnd.Next(PositionsWithoutCenter.Count)];
        }

        public PointF Center()
        {
            return Positions[4];
        }
    }
}