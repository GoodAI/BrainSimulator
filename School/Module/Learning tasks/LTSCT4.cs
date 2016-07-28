using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.IO;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT4 - approach food")]
    public class Ltsct4 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\SCT4\"; }
        }

        public Ltsct4() : this(null) { }

        public Ltsct4(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1},
                { TSHintAttributes.IMAGE_NOISE_BLACK_AND_WHITE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;

            return true;
        }

        protected override void CreateScene()
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            int positionsCount = Positions.Positions.Count;
            int randomLocationIdx = m_rndGen.Next(positionsCount);
            PointF location = Positions.Positions[randomLocationIdx];

            int randomLocationIdx2 = (m_rndGen.Next(positionsCount - 1) + randomLocationIdx + 1) % positionsCount;
            PointF location2 = Positions.Positions[randomLocationIdx2];

            if (m_rndGen.Next(ScConstants.numShapes+1) > 0)
            {
                WrappedWorld.CreateRandomFood(location, size, m_rndGen);
            }
            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                WrappedWorld.CreateRandomStone(location2, size, m_rndGen);
            }
        }
    }
}
