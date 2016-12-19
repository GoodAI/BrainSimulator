using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT7 - approach very good food")]
    public class Ltsct7 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D1\SCT7\"; }
        }

        public Ltsct7() : this(null) { }

        public Ltsct7(SchoolWorld w)
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

            int positionsCount = Positions.Positions.Count;
            int randomLocationIdx = RndGen.Next(positionsCount);
            PointF location = Positions.Positions[randomLocationIdx];

            int randomLocationIdx2 = (RndGen.Next(positionsCount - 1) + randomLocationIdx + 1) % positionsCount;
            PointF location2 = Positions.Positions[randomLocationIdx2];

            Shape randomVeryGoodFood = WrappedWorld.CreateRandomVeryGoodFood(location, size, RndGen);
            Actions.Movement = MoveActionsToTarget(randomVeryGoodFood.GetCenter());

            if (RndGen.Next(3) > 0)
            {
                Shape randomEnemy = WrappedWorld.CreateRandomEnemy(location2, size, RndGen);
                Actions.Movement = NegateMoveActions(MoveActionsToTarget(randomEnemy.GetCenter()));
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
