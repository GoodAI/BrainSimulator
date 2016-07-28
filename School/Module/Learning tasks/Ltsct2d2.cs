using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D2 LT2 - 2 shapes")]
    public class Ltsct2d2 : Ltsct1
    {

        public override string Path
        {
            get { return @"D:\summerCampSamples\D2\SCT2\"; }
        }

        public Ltsct2d2() : this(null) { }

        public Ltsct2d2(SchoolWorld w)
            : base(w)
        {
        }

        protected override void CreateScene()
        {
            Actions = new AvatarsActions(true,false,false,false);

            if (RndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape();
                Actions.Shapes[ShapeIndex] = true;
            }

            if (RndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape();
                Actions.Shapes[ShapeIndex] = true;
            }

            WriteActions();
        }
    }
}
