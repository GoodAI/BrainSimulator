using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D2 LT8 - eat very good food")]
    public class Ltsct8d2 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D2\SCT8\"; }
        }

        public Ltsct8d2() : this(null) { }

        public Ltsct8d2(SchoolWorld w)
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

            PointF location = Positions.Center();

            WrappedWorld.CreateRandomVeryGoodFood(location, size, RndGen);
            Actions.Eat = true;

            PointF location2 = WrappedWorld.RandomPositionInsidePowNonCovering(RndGen, size);
            if (RndGen.Next(3) > 0)
            {
                Shape randomEnemy = WrappedWorld.CreateRandomEnemy(location2, size, RndGen);
                Actions.Eat = false;
                Actions.Movement = NegateMoveActions(MoveActionsToTarget(randomEnemy.Center()));
            }
            else if (RndGen.Next(3) > 0)
            {
                WrappedWorld.CreateRandomFood(location2, size, RndGen);
            }
            else if (RndGen.Next(3) > 0)
            {
                WrappedWorld.CreateRandomStone(location2, size, RndGen);
            }

            WriteActions();
        }
    }
}
