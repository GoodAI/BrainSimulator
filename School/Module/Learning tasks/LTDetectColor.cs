using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Categorize object colors")]
    public class LTDetectColor : AbstractLearningTask<RoguelikeWorld>
    {
        private static readonly TSHintAttribute NUMBER_OF_COLORS = new TSHintAttribute("Number of colors", "", typeof(int), 2, 8);

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
        }

        public override void PresentNewTrainingUnit()
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
            m_target.IsBitmapAsMask = true;

            m_colorIndex = m_rndGen.Next((int)TSHints[NUMBER_OF_COLORS]);
            Color color = LearningTaskHelpers.GetVisibleColor(m_colorIndex);
            m_target.ColorMask = Color.FromArgb(
                AddRandomColorOffset(color.R),
                AddRandomColorOffset(color.G),
                AddRandomColorOffset(color.B));
        }

        protected byte AddRandomColorOffset(byte colorComponent)
        {
            const int MAX_RANDOM_OFFSET = 10;
            return (byte)Math.Max(0, Math.Min(255, colorComponent + m_rndGen.Next(-MAX_RANDOM_OFFSET, MAX_RANDOM_OFFSET + 1)));
        }

        protected void CreateTarget()
        {
            const int TARGET_SIZE = 64;

            m_target = new Shape(Shape.Shapes.Square, PointF.Empty, new SizeF(TARGET_SIZE, TARGET_SIZE));
            WrappedWorld.AddGameObject(m_target);
            // Viewport is assumed to be centered
            float minX = (WrappedWorld.Scene.Width - WrappedWorld.Viewport.Width) / 2;
            float maxX = (WrappedWorld.Scene.Width + WrappedWorld.Viewport.Width) / 2 - m_target.Size.Width;
            m_target.Position.X = (float)(minX + m_rndGen.NextDouble() * (maxX - minX));
            float minY = (WrappedWorld.Scene.Height - WrappedWorld.Viewport.Height) / 2;
            float maxY = (WrappedWorld.Scene.Height + WrappedWorld.Viewport.Height) / 2 - m_target.Size.Height;
            m_target.Position.Y = (float)(minY + m_rndGen.NextDouble() * (maxY - minY));
        }

        public override bool Solve(bool successfully)
        {
            bool didGetCorrectReward = false;

            // Set controls to zero; if solving successfully, set the correct one to 1
            SchoolWorld.ActionInput.Fill(0);
            if (successfully && SchoolWorld.ActionInput.Count > m_colorIndex)
            {
                SchoolWorld.ActionInput.Host[m_colorIndex] = 1;
                SchoolWorld.ActionInput.SafeCopyToDevice();
            }

            // TODO check reward

            return didGetCorrectReward;
        }
    }
}
