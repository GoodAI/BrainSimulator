using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT4")]
    public class Ltsct4 : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly Random m_rndGen = new Random();
        private ScFixPositions m_positions;
        private ScFixColors m_colors;

        public Ltsct4() : this(null) { }

        public Ltsct4(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1}
            };

            TSProgression.Add(TSHints.Clone());
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

            if (m_rndGen.Next(9) == 0) return;

            WrappedWorld.CreateNonVisibleAgent();
            CreateTarget();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;

            return true;
        }

        
        protected void CreateTarget()
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            int positionsCount = m_positions.Positions.Count;
            int randomLocationIdx = m_rndGen.Next(positionsCount);
            PointF location = m_positions.Positions[randomLocationIdx];

            int randomLocationIdx2 = (m_rndGen.Next(positionsCount - 1) + randomLocationIdx + 1) % positionsCount;
            PointF location2 = m_positions.Positions[randomLocationIdx2];

            if (m_rndGen.Next(8) > 0)
            {
                WrappedWorld.CreateRandomFood(location, size, m_rndGen);
            }
            if (m_rndGen.Next(8) > 0)
            {
                WrappedWorld.CreateRandomStone(location2, size, m_rndGen);
            }
        }
    }
}
