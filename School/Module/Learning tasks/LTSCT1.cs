using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT1")]
    public class Ltsct1 : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly Random m_rndGen = new Random();
        bool[][] generationsCheckTable = new bool[9][];
        private ScFixPositions m_positions;
        private ScFixColors m_colors;

        public Ltsct1() : this(null) { }

        public Ltsct1(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1}
            };

            TSProgression.Add(TSHints.Clone());

            for (int i = 0; i < generationsCheckTable.Length; i++)
            {
                generationsCheckTable[i] = new bool[8];
            }
        }

        private bool m_init = true;
        public override void PresentNewTrainingUnit()
        {
            if (m_init)
            {
                m_positions = new ScFixPositions(WrappedWorld.GetPowGeometry());
                m_colors = new ScFixColors(4, WrappedWorld.BackgroundColor);
                m_init = false;
            }

            if (m_rndGen.Next(8) == 1) return;

            WrappedWorld.CreateNonVisibleAgent();
            CreateTarget();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;

            if (!generationsCheckTable.Any(b => b.Any(b1 => b1 == false)))
            {
                MyLog.INFO.WriteLine("Set Is Complete!");
                //wasUnitSuccessful = true;
            }

            return true;
        }

        
        protected void CreateTarget()
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            Color color = m_colors.GetRandomColor(m_rndGen);

            int randomLocationIdx = m_rndGen.Next(m_positions.Positions.Count);
            PointF location = m_positions.Positions[randomLocationIdx];

            int randomShapeIdx = m_rndGen.Next(8);
            Shape.Shapes randomShape = (Shape.Shapes) randomShapeIdx;

            WrappedWorld.CreateShape(randomShape, color, location, size);

            generationsCheckTable[randomLocationIdx][randomShapeIdx] = true;
        }
    }
}
