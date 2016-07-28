using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT5")]
    public class Ltsct5 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        protected override string Path
        {
            get { return @"D:\summerCampSamples\SCT2.csv"; }
        }

        public Ltsct5() : this(null) { }

        public Ltsct5(SchoolWorld w)
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

            PointF location = Positions.Center();

            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                WrappedWorld.CreateRandomFood(location, size, m_rndGen);
            }
            else
            {
                WrappedWorld.CreateRandomStone(location, size, m_rndGen);
            }
        }
    }
}
