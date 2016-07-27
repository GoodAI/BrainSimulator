using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT1")]
    public class Ltsct1 : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly Random m_rndGen = new Random();
        private GameObject m_target;
        private ScFixPositions positions;

        public Ltsct1() : this(null) { }

        public Ltsct1(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.IMAGE_TEXTURE_BACKGROUND, 0);
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

            positions = new ScFixPositions(WrappedWorld.GetPowGeometry());
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            bool wasTargetDetected = Math.Abs(SchoolWorld.ActionInput.Host[0]) > 0.001f;
            bool isTargetPresent = m_target != null;
            wasUnitSuccessful = (wasTargetDetected == isTargetPresent);
            return true;
        }

        protected void CreateTarget()
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width / 4, WrappedWorld.GetPowGeometry().Height / 4);

            PointF location = positions.GetRandomPosition(m_rndGen);
            m_target = WrappedWorld.CreateShape(Shape.GetRandomShape(m_rndGen, 8), Color.White, location, size);
            // Plumber:
            //m_target.X = m_rndGen.Next(0, World.Scene.Width - m_target.Width + 1);
            //m_target.Y = World.Scene.Height - m_target.Height;
            // Roguelike:
            //m_target.X = m_rndGen.Next(0, WrappedWorld.Viewport.Width - m_target.Width + 1);
            //m_target.Y = m_rndGen.Next(0, WrappedWorld.Viewport.Height - m_target.Height + 1);
        }
    }
}
