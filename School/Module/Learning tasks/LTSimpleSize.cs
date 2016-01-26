using System;
using System.Drawing;
using System.Diagnostics;
using GoodAI.Modules.School.Common;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleSize : AbstractLearningTask<ManInWorld>
    {
        public const string COLOR_PATTERNS = "Color patterns";
        public const string TARGET_SIZE_LEVELS = "Target size levels";

        private Random m_rndGen = new Random();
        private GameObject m_agent;
        private GameObject m_target;
        private int m_stepsSincePresented = 0;
        private int m_size = 0; // ranging from 0 to 1; 0-0.125 is smallest, 0.875-1 is biggest; m_size is lower bound of the interval
        private int m_reportedSize = -1;

        public LTSimpleSize(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { COLOR_PATTERNS, 0 },
                { TARGET_SIZE_LEVELS, 2 },
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 1 },
                { TSHintAttributes.NOISE, 0 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_TARGET_DISTANCE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(new TrainingSetHints { 
                { COLOR_PATTERNS, 1 }, 
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 0 }
            });
            TSProgression.Add(new TrainingSetHints { 
                { TARGET_SIZE_LEVELS, 4 }, 
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 2 } 
            });
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000);
            TSProgression.Add(new TrainingSetHints { 
                { TARGET_SIZE_LEVELS, 8 }, 
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 3 }, 
                { TSHintAttributes.MAX_TARGET_DISTANCE, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100 }
            });
            TSProgression.Add(TSHintAttributes.NOISE, 1);

            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            World.FreezeWorld(true);

            CreateAgent();
            CreateTarget();

            m_stepsSincePresented = 0;
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;

            // require immediate decision - in a single step
            if (m_reportedSize >= m_size && m_reportedSize < m_size + 1.0f / TSHints[LTSimpleSize.TARGET_SIZE_LEVELS])
            {
                wasUnitSuccessful = true;
                return true;
            }

            wasUnitSuccessful = false;
            return true;
        }

        protected void CreateAgent()
        {
            World.CreateAgent(null, 0, 0);
            m_agent = World.Agent;
            // center the agent
            m_agent.X = World.FOW_WIDTH / 2 - m_agent.Width / 2;
            m_agent.Y = World.FOW_HEIGHT / 2 - m_agent.Height / 2;
        }

        // scale and position the target:
        protected void CreateTarget()
        {
            // the number of different sizes depends on level:
            int maxWidth = (int)(World.POW_WIDTH * 0.9);
            int maxHeight = (int)(World.POW_HEIGHT * 0.9);
            float fRatio = m_rndGen.Next(1, (int)TSHints[LTSimpleSize.TARGET_SIZE_LEVELS] + 1) / (float)TSHints[LTSimpleSize.TARGET_SIZE_LEVELS];

            int width = (int)(maxWidth * fRatio);
            int height = (int)(maxHeight * fRatio);

            switch (m_rndGen.Next(0, (int)TSHints[TSHintAttributes.TARGET_IMAGE_VARIABILITY]))
            {
                case 0:
                    m_target = new GameObject(GameObjectType.None, @"White10x10.png", m_agent.X, m_agent.Y, width, height);
                    break;
                case 1:
                default:
                    m_target = new GameObject(GameObjectType.None, @"WhiteCircle50x50.png", m_agent.X, m_agent.Y, width, height);
                    break;

            }
            m_target.isBitmapAsMask = TSHints[LTSimpleSize.COLOR_PATTERNS] != 0;
            m_target.maskColor = World.BackgroundColor;
            LearningTaskHelpers.RandomizeColorWDiff(ref m_target.maskColor, 0.1f, m_rndGen);

            World.AddGameObject(m_target);

            // first implementation, random object position (left or right)
            int maxTargetDistanceX = m_rndGen.Next(0, (int)((TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] / 2.0f) * (World.POW_WIDTH - m_target.Width)));
            int maxTargetDistanceY = m_rndGen.Next(0, (int)((TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] / 2.0f) * (World.POW_HEIGHT - m_target.Height)));

            bool isLeft = m_rndGen.Next(0, 2) == 1;
            if (isLeft)
                maxTargetDistanceX = -maxTargetDistanceX;

            bool isUp = m_rndGen.Next(0, 2) == 1;
            if (isUp)
                maxTargetDistanceY = -maxTargetDistanceY;

            m_target.X += maxTargetDistanceX;
            m_target.Y += maxTargetDistanceY;

            // center:
            m_target.X -= (m_target.Width) / 2;
            m_target.Y -= (m_target.Height) / 2;
        }

    }

}
