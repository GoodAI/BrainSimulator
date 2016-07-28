using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT3 - 1 color")]
    public class Ltsct3 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D1\SCT3\"; }
        }

        private readonly Random m_rndGen = new Random();

        public Ltsct3() : this(null) { }

        public Ltsct3(SchoolWorld w)
            : base(w)
        {
        }

        public override void InitCheckTable()
        {
            GenerationsCheckTable = new bool[ScConstants.numPositions + 1][];

            for (int i = 0; i < GenerationsCheckTable.Length; i++)
            {
                GenerationsCheckTable[i] = new bool[ScConstants.numColors];
            }
        }

        protected override void CreateScene()
        {
            Actions = new AvatarsActions(false,true,false,false);

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);
                AddShape(randomLocationIdx);
                Actions.Colors[ColorIndex] = true;
            }

            WriteActions();
        }

        protected override void AddShape(int randomLocationIndex)
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

            Color color = Colors.GetRandomColor(m_rndGen, out ColorIndex);

            PointF location = Positions.Positions[randomLocationIndex];

            ShapeIndex = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)ShapeIndex;

            WrappedWorld.CreateShape(randomShape, color, location, size);

            GenerationsCheckTable[randomLocationIndex][ColorIndex] = true;
        }
    }
}
