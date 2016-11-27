using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D2 LT6 - avoid enemy")]
    public class Ltsct6d2 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D2\SCT6\"; }
        }

        public Ltsct6d2() : this(null) { }

        public Ltsct6d2(SchoolWorld w)
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
            Actions = new AvatarsActions(false, false, true, true);

            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            PointF location = WrappedWorld.RandomPositionInsidePowNonCovering(RndGen, size);

            Shape randomEnemy = WrappedWorld.CreateRandomEnemy(location, size, RndGen);
            Actions.Movement = NegateMoveActions(MoveActionsToTarget(randomEnemy.Center()));

            PointF location2 = WrappedWorld.RandomPositionInsidePowNonCovering(RndGen, size);
            if (LearningTaskHelpers.FlipCoin(RndGen))
            {
                WrappedWorld.CreateRandomFood(location2, size, RndGen);
            }
            else
            {
                WrappedWorld.CreateRandomStone(location2, size, RndGen);
            }

            WriteActions();
        }
    }
}
