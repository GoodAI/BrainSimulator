using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT7 - approach very good food")]
    public class Ltsct7 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        protected override string Path
        {
            get { return @"D:\summerCampSamples\SCT2.csv"; }
        }

        public Ltsct7() : this(null) { }

        public Ltsct7(SchoolWorld w)
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

            WrappedWorld.CreateRandomVeryGoodFood(location, size, m_rndGen);

            if (m_rndGen.Next(3) > 0)
            {
                WrappedWorld.CreateRandomEnemy(location2, size, m_rndGen);
            }
            else if (m_rndGen.Next(3) > 0)
            {
                WrappedWorld.CreateRandomFood(location2, size, m_rndGen);
            }
            else if (m_rndGen.Next(3) > 0)
            {
                WrappedWorld.CreateRandomStone(location2, size, m_rndGen);
            }
        }
    }
}
