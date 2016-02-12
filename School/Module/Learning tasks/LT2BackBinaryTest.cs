using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;
using System.Linq;

namespace GoodAI.Modules.School.LearningTasks
{
    class LT2BackBinaryTest : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected Shape m_target;
        protected Shape.Shapes m_target_type;

        Shape[] m_lastNShapes;

        const int N_BACK = 2;

        public LT2BackBinaryTest() : base(null) { }

        public LT2BackBinaryTest(SchoolWorld w)
            : base(w)
        {
            m_lastNShapes = new Shape[N_BACK + 1];

            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.IS_VARIABLE_COLOR, 0},
                {TSHintAttributes.IS_VARIABLE_SIZE, 0},
                {TSHintAttributes.IS_VARIABLE_POSITION, 0},
                {TSHintAttributes.IS_VARIABLE_ROTATION, 0},
                {TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1f);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1f);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_ROTATION, 1f);
            //TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3f);
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();

            //random size
            Size shapeSize = new Size(120, 120);
            Point shapePosition = new Point(WrappedWorld.FOW_WIDTH / 2, WrappedWorld.FOW_HEIGHT / 2);
            Color shapeColor = Color.White;
            float rotation = 0;

            Shape nthShape = m_lastNShapes[m_lastNShapes.Length - 1];
            if (nthShape != null && m_rndGen.NextDouble() < 0.5)
            {
                // with probability 0.5 copy the same
                shapeSize = new Size(nthShape.Width, nthShape.Height);
                shapePosition.X = nthShape.X;
                shapePosition.Y = nthShape.Y;
                shapeColor = nthShape.maskColor;
                rotation = nthShape.Rotation;
                m_target_type = nthShape.ShapeType;
            }
            else
            {
                // with probability 0.5 create a random new one

                // generate random size
                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1.0f)
                {
                    int side = m_rndGen.Next(60, 121);
                    shapeSize = new Size(side, side);
                }

                // random position
                shapePosition.X -= shapeSize.Width / 2;
                shapePosition.Y -= shapeSize.Height / 2;

                if (TSHints[TSHintAttributes.IS_VARIABLE_POSITION] >= 1.0f)
                {
                    shapePosition = WrappedWorld.RandomPositionInsidePow(m_rndGen, shapeSize);
                }

                // random color
                if (TSHints[TSHintAttributes.IS_VARIABLE_COLOR] >= 1.0f)
                {
                    shapeColor = LearningTaskHelpers.RandomVisibleColor(m_rndGen);
                }

                // random rotation
                if (TSHints[TSHintAttributes.IS_VARIABLE_ROTATION] >= 1.0f)
                {
                    rotation = (float)(m_rndGen.NextDouble() * 360);
                }

                // random shape
                m_target_type = Shape.GetRandomShape(m_rndGen, (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS]);
            }

            m_target = (Shape)WrappedWorld.CreateShape(shapePosition, m_target_type, shapeColor, shapeSize, rotation: rotation);

            push(m_target);
        }

        void push(Shape pushedObject)
        {
            for (int i = m_lastNShapes.Length - 1; i > 0; i--)
            {
                m_lastNShapes[i] = m_lastNShapes[i - 1];
            }
            m_lastNShapes[0] = pushedObject;
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            bool shapeEqual = shapesEqual(m_lastNShapes[0], m_lastNShapes[m_lastNShapes.Length - 1]);    // shapes are equal

            bool responseYes = WrappedWorld.Controls.Host[0] >= 1;            // agent response >= 1, which means YES
            bool responseNo = WrappedWorld.Controls.Host[0] <= 0;             // agent response <= 0, which means NO

            if (
                (shapeEqual && responseYes)
                ||
                (!shapeEqual && responseNo))
            {
                wasUnitSuccessful = true;
            }
            else
            {
                wasUnitSuccessful = false;
            }
            return true;
        }

        bool shapesEqual(Shape s1, Shape s2)
        {
            if (s1 == null && s2 == null)
            {
                return true;
            }
            else if (s1 == null || s2 == null)
            {
                return false;
            }

            bool shapeTypes = s1.ShapeType.Equals(s1.ShapeType);
            bool colors = s1.maskColor == s2.maskColor;
            bool positions = s1.X == s2.X && s1.Y == s2.Y;
            bool sizes = s1.Width == s2.Width && s1.Height == s2.Height;
            bool rotations = s1.Rotation == s2.Rotation;

            return shapeTypes && colors && positions && sizes && rotations;
        }
    }
}
