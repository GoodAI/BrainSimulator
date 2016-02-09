using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{

    public class LTDetectColor : AbstractLearningTask<RoguelikeWorld>
    {
        private static readonly TSHintAttribute NUMBER_OF_COLORS = new TSHintAttribute("Condition salience", "", typeof(int), 0, 8);

        protected GameObject m_target;
        protected Random m_rndGen = new Random();

        protected int m_colorIndex;

        public LTDetectColor() : this(null) { }

        public LTDetectColor(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { NUMBER_OF_COLORS, 2 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(NUMBER_OF_COLORS, 4);
            TSProgression.Add(NUMBER_OF_COLORS, 8);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            CreateTarget();
            SetTargetColor();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // Identify color with a 1-of-k encoding (or confidence values)
            int arraySize = Math.Min(SchoolWorld.ActionInput.Count, (int)TSHints[NUMBER_OF_COLORS]);
            int guessedColor = GetMaxIndex(SchoolWorld.ActionInput.Host, arraySize);

            wasUnitSuccessful = guessedColor == m_colorIndex;
            return true;
        }

        protected int GetMaxIndex(float[] array, int numberOfElements)
        {
            float max = float.NegativeInfinity;
            int maxIndex = -1;
            for (int index = 0; index < numberOfElements; index++)
            {
                if (array[index] > max)
                {
                    max = array[index];
                    maxIndex = index;
                }
            }
            return maxIndex;
        }

        protected void SetTargetColor()
        {
            m_target.isBitmapAsMask = true;

            m_colorIndex = m_rndGen.Next((int)TSHints[NUMBER_OF_COLORS]);
            Color color = LearningTaskHelpers.GetVisibleColor(m_colorIndex);
            m_target.maskColor = Color.FromArgb(
                AddRandomColorOffset(color.R),
                AddRandomColorOffset(color.G),
                AddRandomColorOffset(color.B));
        }

        protected byte AddRandomColorOffset(byte colorComponent)
        {
            const int MAX_RANDOM_OFFSET = 10;
            return (byte)Math.Max(0, Math.Min(255,
                (int)colorComponent + m_rndGen.Next(-MAX_RANDOM_OFFSET, MAX_RANDOM_OFFSET + 1)));
        }

        protected void CreateTarget()
        {
            m_target = new Shape(Shape.Shapes.Square, 0, 0);
            WrappedWorld.AddGameObject(m_target);
            // POW is assumed to be centered
            int minX = (WrappedWorld.FOW_WIDTH - WrappedWorld.POW_WIDTH) / 2;
            int maxX = (WrappedWorld.FOW_WIDTH + WrappedWorld.POW_WIDTH) / 2 - m_target.Width;
            m_target.X = m_rndGen.Next(minX, maxX + 1);
            int minY = (WrappedWorld.FOW_HEIGHT - WrappedWorld.POW_HEIGHT) / 2;
            int maxY = (WrappedWorld.FOW_HEIGHT + WrappedWorld.POW_HEIGHT) / 2 - m_target.Height;
            m_target.Y = m_rndGen.Next(minY, maxY + 1);
        }

    }
}
