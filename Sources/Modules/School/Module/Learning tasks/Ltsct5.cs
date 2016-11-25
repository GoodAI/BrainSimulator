using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT5 - eat food")]
    public class Ltsct5 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D1\SCT5\"; }
        }

        public Ltsct5() : this(null) { }

        public Ltsct5(SchoolWorld w)
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
            Actions = new AvatarsActions(false,false,true,true);

            if (RndGen.Next(10) > 0)
            {
                SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

                PointF location = Positions.Center();

                if (LearningTaskHelpers.FlipCoin(RndGen))
                {
                    WrappedWorld.CreateRandomFood(location, size, RndGen);
                    Actions.Eat = true;
                }
                else
                {
                    WrappedWorld.CreateRandomStone(location, size, RndGen);
                }
                if (LearningTaskHelpers.FlipCoin(RndGen))
                {
                    PointF location2 = Positions.GetRandomPositionWithoutCenter(RndGen);
                    WrappedWorld.CreateRandomStone(location2, size, RndGen);
                }
            }

            WriteActions();
        }
    }
}
