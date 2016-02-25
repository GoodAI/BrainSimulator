using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Detect object presence")]
    public class LTDetectWhite : AbstractLearningTask<RoguelikeWorld>
    {
        protected readonly Random m_rndGen = new Random();
        protected GameObject m_target;

        public LTDetectWhite() : this(null) { }

        public LTDetectWhite(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
        }

        public override void PresentNewTrainingUnit()
        {
            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                WrappedWorld.CreateNonVisibleAgent();
                CreateTarget();
            }
            else
            {
                m_target = null;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            bool wasTargetDetected = SchoolWorld.ActionInput.Host[0] != 0;
            bool isTargetPresent = m_target != null;
            wasUnitSuccessful = (wasTargetDetected == isTargetPresent);

            return true;
        }

        protected void CreateTarget()
        {
            const int TARGET_SIZE = 32;
            SizeF size = new SizeF(TARGET_SIZE, TARGET_SIZE);
            PointF location = WrappedWorld.RandomPositionInsideViewport(m_rndGen, size);
            m_target = WrappedWorld.CreateGameObject(@"White10x10.png", location, size);
            // Plumber:
            //m_target.X = m_rndGen.Next(0, World.Scene.Width - m_target.Width + 1);
            //m_target.Y = World.Scene.Height - m_target.Height;
            // Roguelike:
            //m_target.X = m_rndGen.Next(0, WrappedWorld.Viewport.Width - m_target.Width + 1);
            //m_target.Y = m_rndGen.Next(0, WrappedWorld.Viewport.Height - m_target.Height + 1);
        }
    }
}
