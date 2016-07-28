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
    [DisplayName("SC D2 LT1 - 1 shape")]
    public class Ltsct1d2 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();
        public override string Path { get { return @"D:\summerCampSamples\D2\SCT1\"; } }

        public Ltsct1d2() : this(null) { }

        public Ltsct1d2(SchoolWorld w)
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
            Actions = new AvatarsActions();

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape();

                Actions.Shapes[ShapeIndex] = true;
            }

            WriteActions();
        }

        protected void AddShape()
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            Color color = Colors.GetRandomColor(m_rndGen);

            PointF location = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, size);

            ShapeIndex = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)ShapeIndex;

            WrappedWorld.CreateShape(randomShape, color, location, size);
        }
    }
}
