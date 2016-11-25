using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Categorize shape or color")]
    public class LTDetectShapeColor : AbstractLearningTask<ManInWorld>
    {
        private readonly TSHintAttribute COLOR_COUNT = new TSHintAttribute("Number of different colors.", "The color pool used to pick a shape's color.", typeof(int), 0, 1); //check needed;

        private int[,] m_classes;
        protected readonly Random m_rndGen = new Random();
        protected Shape m_target;
        private int m_colorIdx;

        public LTDetectShapeColor() : this(null) { }

        public LTDetectShapeColor(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000},
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.IS_VARIABLE_SIZE, 0},
                {TSHintAttributes.IS_VARIABLE_POSITION, 0},
                {TSHintAttributes.IS_VARIABLE_ROTATION, 0},
                {TSHintAttributes.DEGREES_OF_FREEDOM, 2},
                {TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2},
                {COLOR_COUNT, 2},
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1);
            TSProgression.Add(COLOR_COUNT, 3);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 3);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 4);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_ROTATION, 1);
            TSProgression.Add(COLOR_COUNT, 4);
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 4);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 5);
            TSProgression.Add(COLOR_COUNT, 5);

            SetClasses();
        }

        public override void IncreaseLevel()
        {
            base.IncreaseLevel();

            SetClasses();
        }

        void SetClasses()
        {
            int shapeCount = (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS];
            int colorCount = (int)TSHints[COLOR_COUNT];
            m_classes = new int[shapeCount, colorCount];

            int classCount = (int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM];

            for (int j = 0; j < colorCount; j++)
                for (int i = 0; i < shapeCount; i++)
                    m_classes[i, j] = m_rndGen.Next(classCount);
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();

            // wtih Pr=.5 show object
            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                // random size
                SizeF shapeSize = new SizeF(32, 32);
                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1.0f)
                {
                    float side = (float)(10 + m_rndGen.NextDouble() * 38);
                    shapeSize = new SizeF(side, side);
                }

                // random position
                PointF shapePosition = WrappedWorld.Agent.GetGeometry().Location + new Size(20, 0);
                if (TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1.0f)
                {
                    shapePosition = WrappedWorld.RandomPositionInsideViewport(m_rndGen, shapeSize, 2);
                }

                // random color
                m_colorIdx = m_rndGen.Next((int)TSHints[COLOR_COUNT]);
                Color shapeColor = LearningTaskHelpers.GetVisibleColor(m_colorIdx);

                // random rotation
                float rotation = 0;
                if (TSHints[TSHintAttributes.IS_VARIABLE_ROTATION] >= 1.0f)
                {
                    rotation = (float)(m_rndGen.NextDouble() * 360);
                }

                // random shape
                Shape.Shapes shape = Shape.GetRandomShape(m_rndGen, (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]);

                m_target = WrappedWorld.CreateShape(shape, shapeColor, shapePosition, shapeSize, rotation);
            }
            else
            {
                m_target = null;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            int classCount = (int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM];

            // no target
            if (m_target == null)
            {
                // Fail if there is any agent action
                wasUnitSuccessful = false;
                return !WrappedWorld.Controls.Host.Take(classCount).Any(a => a > 0);
            }

            // Get the highest signal in agent's outputs
            int idxOfMax = WrappedWorld.Controls.Host.Take(classCount)
                .Select((x, i) => new { Value = x, Index = i })
                .Aggregate(
                    new { Value = float.MinValue, Index = -1 },
                    (a, x) => (a.Index < 0) || (x.Value > a.Value) ? x : a,
                    a => a.Index
                );

            wasUnitSuccessful = m_classes[(int)m_target.ShapeType, m_colorIdx] == idxOfMax;
            return true;
        }
    }
}
