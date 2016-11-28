using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT1 - 1 shape - fixed position")]
    public class Ltsct1Fp : Ltsct1
    {
        public override string Path { get { return @"D:\summerCampSamples\D1\SCT1FP\"; } }

        public Ltsct1Fp() : this(null) { }

        public Ltsct1Fp(SchoolWorld w)
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
            Actions = new AvatarsActions(true,false,false,false);

            if (RndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                const int fixedLocationIndex = 4;
                AddShape(fixedLocationIndex);

                Actions.Shapes[ShapeIndex] = true;
            }

            WriteActions();
        }
    }
}
