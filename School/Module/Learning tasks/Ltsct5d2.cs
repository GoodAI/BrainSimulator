using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D2 LT5 - eat food")]
    public class Ltsct5d2 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\D2\SCT5\"; }
        }

        public Ltsct5d2() : this(null) { }

        public Ltsct5d2(SchoolWorld w)
            : base(w)
        {
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = true;

            return true;
        }

        
        protected override void CreateScene()
        {
            Actions = new AvatarsActions();

            if (m_rndGen.Next(9) > 0)
            {
                SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

                PointF location = Positions.Center();

                if (LearningTaskHelpers.FlipCoin(m_rndGen))
                {
                    WrappedWorld.CreateRandomFood(location, size, m_rndGen);
                    Actions.Eat = true;
                }
                else
                {
                    WrappedWorld.CreateRandomStone(location, size, m_rndGen);
                }

                if (LearningTaskHelpers.FlipCoin(m_rndGen))
                {
                    PointF location2 = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, size);
                    WrappedWorld.CreateRandomStone(location2, size, m_rndGen);
                }
            }

            WriteActions();
        }
    }
}
