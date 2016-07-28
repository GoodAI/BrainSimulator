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
    [DisplayName("SC D2 LT4 - approach food")]
    public class Ltsct4d2 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\D2\SCT4\"; }
        }

        public Ltsct4d2() : this(null) { }

        public Ltsct4d2(SchoolWorld w)
            : base(w)
        {
        }

        protected override void CreateScene()
        {
            Actions = new AvatarsActions();

            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            PointF location = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, size);

            PointF location2 = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, size);

            if (m_rndGen.Next(ScConstants.numShapes+1) > 0)
            {
                Shape randomFood = WrappedWorld.CreateRandomFood(location, size, m_rndGen);
                Actions.Movement = MoveActionsToTarget(randomFood.Center());
            }
            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                WrappedWorld.CreateRandomStone(location2, size, m_rndGen);
            }
            
            Actions.WriteActions(StreamWriter);string joinedActions = Actions.ToString();MyLog.INFO.WriteLine(joinedActions);

        }
    }
}
