using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT3 - 1 color")]
    public class Ltsct3 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public Ltsct3() : this(null) { }

        public Ltsct3(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1},
                { TSHintAttributes.IMAGE_NOISE_BLACK_AND_WHITE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        public override void InitCheckTable()
        {
            generationsCheckTable = new bool[ScConstants.numPositions + 1][];

            for (int i = 0; i < generationsCheckTable.Length; i++)
            {
                generationsCheckTable[i] = new bool[ScConstants.numColors];
            }
        }

        protected override void CreateScene()
        {
            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);
                AddShape(randomLocationIdx);
            }
        }

        protected override void AddShape(int randomLocationIndex)
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            int randomColorIndex;
            Color color = m_colors.GetRandomColor(m_rndGen, out randomColorIndex);

            PointF location = m_positions.Positions[randomLocationIndex];

            int randomShapeIdx = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)randomShapeIdx;

            WrappedWorld.CreateShape(randomShape, color, location, size);

            generationsCheckTable[randomLocationIndex][randomColorIndex] = true;
        }
    }
}
