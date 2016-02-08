using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{

    public class LTDetectColor : AbstractLearningTask<RoguelikeWorld>
    {
        private static readonly TSHintAttribute NUMBER_OF_COLORS = new TSHintAttribute("Condition salience", "", TypeCode.Single, 0, 1); //check needed;

        protected GameObject m_target;
        protected Random m_rndGen = new Random();

        protected int m_colorIndex;

        public LTDetectColor() { }

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
            int guessedColor = GetMaxIndex(SchoolWorld.ActionInput.Host, (int)TSHints[NUMBER_OF_COLORS]);

            wasUnitSuccessful = true;
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


    /*
    public abstract class AbstractWorldAdapter<WorldType> : IWorldAdapter where WorldType : ManInWorld
    {
        protected Random m_rndGen = new Random();
        protected WorldType m_w;

        protected virtual WorldType World
        {
            get { return m_w; }
        }

        protected virtual void InstallWorld(WorldType w, TrainingSetHints trainingSetHints)
        {
            m_w = w;
            m_w.ClearWorld();
            if (trainingSetHints[TSHintAttributes.NOISE] > 0)
            {
                m_w.IsImageNoise = true;
            }
        }

        public virtual void PresentNewTrainingUnit(AbstractSchoolWorld w, TrainingSetHints hints)
        {
            InstallWorld(w as WorldType, hints);
        }

        public virtual bool IsTrainingUnitCompleted(ref bool wasUnitSuccessful)
        {
            if (World.IsEmulatingUnitCompletion())
            {
                return World.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
            }
            else
            {
                wasUnitSuccessful = true;
                return true;
            }
        }
    }*/

    /*
    public abstract class AbstractWADetectColor : IWorldAdapter
    {
        protected abstract AbstractSchoolWorld World { get; }


        public void PresentNewTrainingUnit(AbstractSchoolWorld w, TrainingSetHints hints)
        {
            InstallWorld(w, hints);
            CreateTarget(hints);
            SetTargetColor();
        }


        protected abstract void InstallWorld(AbstractSchoolWorld w, TrainingSetHints trainingSetHints);
        protected abstract void CreateTarget(TrainingSetHints trainingSetHints);
    }

    public class PlumberWorldWADetectColor : AbstractWorldAdapter<PlumberWorld>
    {
        protected GameObject m_target;

        public override void PresentNewTrainingUnit(AbstractSchoolWorld w, TrainingSetHints hints)
        {
            CreateTarget(hints);
            SetTargetColor();
        }

        protected void SetTargetColor()
        {
            m_target.isBitmapAsMask = true;
            LearningTaskHelpers.RandomizeColor(ref m_target.maskColor, m_rndGen);
        }

        protected void CreateTarget(TrainingSetHints trainingSetHints)
        {
            m_target = new GameObject(GameObjectType.None, @"White10x10.png", 0, 0);
            m_w.AddGameObject(m_target);
            // POW is assumed to be centered
            int minX = (m_w.FOW_WIDTH - m_w.POW_WIDTH) / 2;
            int maxX = (m_w.FOW_WIDTH + m_w.POW_WIDTH) / 2 - m_target.Width;
            m_target.X = m_rndGen.Next(minX, maxX + 1);
            m_target.Y = (m_w.FOW_HEIGHT + m_w.POW_HEIGHT) / 2 - m_target.Height;
        }
    }

    public class RoguelikeWorldWADetectColor : AbstractWorldAdapter<RoguelikeWorld>
    {
        protected GameObject m_target;

        protected void CreateTarget(TrainingSetHints trainingSetHints)
        {
            m_target = new GameObject(GameObjectType.None, @"White10x10.png", 0, 0);
            m_w.AddGameObject(m_target);
            // POW is assumed to be centered
            int minX = (m_w.FOW_WIDTH - m_w.POW_WIDTH) / 2;
            int maxX = (m_w.FOW_WIDTH + m_w.POW_WIDTH) / 2 - m_target.Width;
            m_target.X = m_rndGen.Next(minX, maxX + 1);
            int minY = (m_w.FOW_HEIGHT - m_w.POW_HEIGHT) / 2;
            int maxY = (m_w.FOW_HEIGHT + m_w.POW_HEIGHT) / 2 - m_target.Height;
            m_target.Y = m_rndGen.Next(minY, maxY + 1);
        }
    }
    */
}
