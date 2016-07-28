using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D2 LT3 - 1 color")]
    public class Ltsct3d2 : Ltsct1
    {
        public override string Path
        {
            get { return @"D:\summerCampSamples\D2\SCT3\"; }
        }

        private readonly Random m_rndGen = new Random();

        public Ltsct3d2() : this(null) { }

        public Ltsct3d2(SchoolWorld w)
            : base(w)
        {
        }

        public override void InitCheckTable()
        {
            GenerationsCheckTable = new bool[ScConstants.numPositions + 1][];

            for (int i = 0; i < GenerationsCheckTable.Length; i++)
            {
                GenerationsCheckTable[i] = new bool[ScConstants.numColors];
            }
        }

        protected override void CreateScene()
        {
            Actions = new AvatarsActions();

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape();
                Actions.Colors[ColorIndex] = true;
            }

            Actions.WriteActions(StreamWriter);string joinedActions = Actions.ToString();MyLog.INFO.WriteLine(joinedActions);
        }
    }
}
