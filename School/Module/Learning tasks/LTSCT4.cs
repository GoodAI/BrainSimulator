using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT4 - approach food")]
    public class Ltsct4 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\D1\SCT4\"; }
        }

        public Ltsct4() : this(null) { }

        public Ltsct4(SchoolWorld w)
            : base(w)
        {
        }

        protected override void CreateScene()
        {
            Actions = new AvatarsActions(false,false,true,true);

            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            int positionsWcCount = Positions.PositionsWithoutCenter.Count;
            int randomLocationIdx = m_rndGen.Next(positionsWcCount);
            PointF location = Positions.PositionsWithoutCenter[randomLocationIdx];

            int randomLocationIdx2 = (m_rndGen.Next(positionsWcCount - 1) + randomLocationIdx + 1) % positionsWcCount;
            PointF location2 = Positions.PositionsWithoutCenter[randomLocationIdx2];

            if (m_rndGen.Next(ScConstants.numShapes+1) > 0)
            {
                Shape randomFood = WrappedWorld.CreateRandomFood(location, size, m_rndGen);
                Actions.Movement = MoveActionsToTarget(randomFood.Center());
            }
            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                WrappedWorld.CreateRandomStone(location2, size, m_rndGen);
            }

            WriteActions();

        }
    }
}
