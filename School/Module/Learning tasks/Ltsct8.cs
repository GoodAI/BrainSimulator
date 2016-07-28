using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT8 - eat very good food")]
    public class Ltsct8 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D1\SCT8\"; }
        }

        public Ltsct8() : this(null) { }

        public Ltsct8(SchoolWorld w)
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
            const int randomLocationIdx = 4;
            PointF location = Positions.Positions[randomLocationIdx];

            int randomLocationIdx2 = (RndGen.Next(positionsCount - 1) + randomLocationIdx + 1) % positionsCount;
            PointF location2 = Positions.Positions[randomLocationIdx2];

            WrappedWorld.CreateRandomVeryGoodFood(location, size, RndGen);
            Actions.Eat = true;

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
