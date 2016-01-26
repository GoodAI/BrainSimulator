using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    // TODO: Currently presents target outside of POW.

    class LTDetectWhite : AbstractLearningTask<ManInWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;

        public LTDetectWhite(ManInWorld w) : base(w)
        {
            TSHints = new TrainingSetHints { 
                { TSHintAttributes.NOISE, 0 }, 
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            World.ClearWorld(TSHints);
            if (LearningTaskHelpers.FlipCoin(m_rndGen))
            {
                CreateTarget();
            }
            else
            {
                m_target = null;
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // We currently use ContinuousControl as the agent actuator.
            // Should refactor to have just one block of actuators.
            bool wasTargetDetected = (World as ManInWorld).Controls.Host[0] != 0;
            bool isTargetPresent = m_target != null;
            wasUnitSuccessful = (wasTargetDetected == isTargetPresent);

            //MyLog.INFO.WriteLine("Unit completed with " + (wasUnitSuccessful ? "success" : "failure"));
            return true;
        }

        protected void CreateTarget()
        {
            m_target = new GameObject(GameObjectType.None, @"White10x10.png", 0, 0);
            World.AddGameObject(m_target);
            // Plumber:
            //m_target.X = m_rndGen.Next(0, World.FOW_WIDTH - m_target.Width + 1);
            //m_target.Y = World.FOW_HEIGHT - m_target.Height;
            // Roguelike:
            m_target.X = m_rndGen.Next(0, World.FOW_WIDTH - m_target.Width + 1);
            m_target.Y = m_rndGen.Next(0, World.FOW_HEIGHT - m_target.Height + 1);
        }
    }

    /*
    public class RoguelikeWorldWADetectWhite : AbstractWADetectWhite
    {
        private Worlds m_w;

        protected override AbstractSchoolWorld World
        {
            get
            {
                return m_w;
            }
        }

        protected override void InstallWorld(AbstractSchoolWorld w, TrainingSetHints trainingSetHints)
        {
            m_w = w as RoguelikeWorld;
            m_w.ClearWorld();
            if (trainingSetHints[TSHintAttributes.NOISE] > 0)
            {
                m_w.IsImageNoise = true;
            }
        }

        protected override void CreateTarget(TrainingSetHints trainingSetHints)
        {
            m_target = new GameObject(GameObjectType.None, @"White10x10.png", 0, 0);
            m_w.AddGameObject(m_target);
            m_target.X = m_rndGen.Next(0, m_w.FOW_WIDTH - m_target.Width + 1);
            m_target.Y = m_rndGen.Next(0, m_w.FOW_HEIGHT - m_target.Height + 1);
        }
    }
*/

}
