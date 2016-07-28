using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT2 - 2 shapes")]
    public class Ltsct2 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\D1\SCT2\"; }
        }

        public Ltsct2() : this(null) { }

        public Ltsct2(SchoolWorld w)
            : base(w)
        {
        }

        protected override void CreateScene()
        {
            Actions = new AvatarsActions();

            int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape(randomLocationIdx);
                Actions.Shapes[ShapeIndex] = true;
            }


            int nextRandomLocationIdx = m_rndGen.Next(randomLocationIdx + 1, randomLocationIdx + ScConstants.numPositions);
            nextRandomLocationIdx %= ScConstants.numPositions;

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape(nextRandomLocationIdx);
                Actions.Shapes[ShapeIndex] = true;
            }


            Actions.WriteActions(StreamWriter);string joinedActions = Actions.ToString();MyLog.INFO.WriteLine(joinedActions);
        }
    }
}
