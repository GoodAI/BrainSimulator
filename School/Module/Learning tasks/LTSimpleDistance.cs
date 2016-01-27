using GoodAI.Modules.School.Common;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTSimpleDistance : DeprecatedAbstractLearningTask<ManInWorld>
    {
        public const string COLOR_PATTERNS = "Color patterns";
        public const string TARGET_SIZE_LEVELS = "Target size levels";
        public const string TARGET_DISTANCE_LEVELS = "Target distance levels";

        private Random m_rndGen = new Random();
        private GameObject m_agent;
        private GameObject m_target;
        private int m_stepsSincePresented = 0;
        private int m_distance = 0; // ranging from 0 to 1; 0-0.125 is smallest, 0.875-1 is biggest; m_distance is lower bound of the interval
        private int m_reportedDistance = -1;

        public LTSimpleDistance() { }

        public LTSimpleDistance(ManInWorld w)
            : base(w)
        {

            TSHints = new TrainingSetHints
            {
                { COLOR_PATTERNS, 0 },
                { TARGET_SIZE_LEVELS, 1 },
                { TARGET_DISTANCE_LEVELS, 2 },
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 1 },
                { TSHintAttributes.NOISE, 0 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            // TODO
            TSProgression.Add(new TrainingSetHints {
                { COLOR_PATTERNS, 1 },
                { TARGET_DISTANCE_LEVELS, 8 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 0 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 2 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TARGET_SIZE_LEVELS, 8 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000 }
            });
            TSProgression.Add(new TrainingSetHints {
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 3 },
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
            if (m_reportedDistance >= m_distance && m_reportedDistance < m_distance + 1.0f / TSHints[TARGET_DISTANCE_LEVELS])
            {
                wasUnitSuccessful = true;
                return true;
            }

            wasUnitSuccessful = false;
            return true;
        }

        private void CreateAgent()
        {
            World.CreateAgent(null, 0, 0);
            m_agent = World.Agent;
            // center the agent
            m_agent.X = World.FOW_WIDTH / 2 - m_agent.Width / 2;
            m_agent.Y = World.FOW_HEIGHT / 2 - m_agent.Height / 2;
        }

        // scale and position the target:
        private void CreateTarget()
        {
            // the number of different sizes depends on level. The minimum size is 0.8*maximum size
            int maxWidth = (int)(World.POW_WIDTH * 0.3);
            int maxHeight = (int)(World.POW_HEIGHT * 0.3);
            float ratio = m_rndGen.Next(1, (int)TSHints[TARGET_SIZE_LEVELS] + 1) / TSHints[TARGET_SIZE_LEVELS];
            float minRatio = 0.8f;

            int width = (int)(maxWidth * minRatio + maxWidth * (1.0f - minRatio) * ratio);
            int height = (int)(maxHeight * minRatio + maxHeight * (1.0f - minRatio) * ratio);

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
            m_target.isBitmapAsMask = TSHints[COLOR_PATTERNS] != 0;
            m_target.maskColor = World.BackgroundColor;
            LearningTaskHelpers.RandomizeColorWDiff(ref m_target.maskColor, 0.1f, m_rndGen);

            World.AddGameObject(m_target);

            // first implementation, random object position (left or right)
            float maxDistanceRatioX = m_rndGen.Next(0, (int)TSHints[TARGET_DISTANCE_LEVELS] + 1) / TSHints[TARGET_DISTANCE_LEVELS];
            float maxDistanceRatioY = m_rndGen.Next(0, (int)TSHints[TARGET_DISTANCE_LEVELS] + 1) / TSHints[TARGET_DISTANCE_LEVELS];
            int maxTargetDistanceX = (int)(maxDistanceRatioX * (World.POW_WIDTH - m_target.Width) / 2.0f);
            int maxTargetDistanceY = (int)(maxDistanceRatioY * (World.POW_HEIGHT - m_target.Height) / 2.0f);

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

    // Abstract base class for world adapters
    //public abstract class AbstractWASimpleDistance : IDeprecatedWorldAdapter
    //{
    //    protected Random m_rndGen = new Random();
    //    protected GameObject m_agent;
    //    protected GameObject m_target;
    //    protected int m_stepsSincePresented = 0;
    //    protected int m_distance = 0; // ranging from 0 to 1; 0-0.125 is smallest, 0.875-1 is biggest; m_distance is lower bound of the interval
    //    protected int m_reportedDistance = -1;
    //    TSHintsSimpleDistance hints;

    //    protected abstract AbstractSchoolWorld World { get; }

    //    public void PresentNewTrainingUnit(AbstractSchoolWorld w, IHints hints)
    //    {
    //        this.hints = hints as TSHintsSimpleDistance;
    //        InstallWorld(w, hints as TSHintsSimpleDistance);
    //        CreateAgent();
    //        CreateTarget(hints as TSHintsSimpleDistance);

    //        m_stepsSincePresented = 0;
    //    }

    //    public bool IsTrainingUnitCompleted(out bool wasUnitSuccessful)
    //    {
    //        if (World.IsEmulatingUnitCompletion())
    //        {
    //            return World.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
    //        }
    //        else
    //        {
    //            // expect this method to be called once per simulation step
    //            m_stepsSincePresented++;

    //            // require immediate decision - in a single step
    //            if (m_reportedDistance >= m_distance && m_reportedDistance < m_distance + 1.0f / hints.TargetDistanceLevels)
    //            {
    //                wasUnitSuccessful = true;
    //                return true;
    //            }

    //            wasUnitSuccessful = false;
    //            return true;
    //        }
    //    }

    //    protected abstract void CreateAgent();
    //    protected abstract void InstallWorld(AbstractSchoolWorld w, TSHintsSimpleDistance trainingSetHints);
    //    protected abstract void CreateTarget(TSHintsSimpleDistance trainingSetHints);
    //}

    //public class ManInWorldWASimpleDistance : AbstractWASimpleDistance
    //{
    //    private ManInWorld m_w;

    //    protected override AbstractSchoolWorld World
    //    {
    //        get
    //        {
    //            return m_w;
    //        }
    //    }

    //    protected override void InstallWorld(AbstractSchoolWorld w, TSHintsSimpleDistance trainingSetHints)
    //    {
    //        m_w = w as ManInWorld;
    //        m_w.ClearWorld();
    //        m_w.FreezeWorld(true);
    //        if (trainingSetHints.Noise > 0)
    //        {
    //            m_w.IsImageNoise = true;
    //        }
    //    }

    //    protected override void CreateAgent()
    //    {
    //        m_w.CreateAgent(null, 0, 0);
    //        m_agent = m_w.Agent;
    //        // center the agent
    //        m_agent.X = m_w.FOW_WIDTH / 2 - m_agent.Width / 2;
    //        m_agent.Y = m_w.FOW_HEIGHT / 2 - m_agent.Height / 2;
    //    }

    //    // scale and position the target:
    //    protected override void CreateTarget(TSHintsSimpleDistance trainingSetHints)
    //    {
    //        // the number of different sizes depends on level. The minimum size is 0.8*maximum size
    //        int maxWidth = (int)(m_w.POW_WIDTH * 0.3);
    //        int maxHeight = (int)(m_w.POW_HEIGHT * 0.3);
    //        float ratio = m_rndGen.Next(1, trainingSetHints.TargetSizeLevels + 1) / (float)trainingSetHints.TargetSizeLevels;
    //        float minRatio = 0.8f;

    //        int width = (int)(maxWidth * minRatio + maxWidth * (1.0f - minRatio) * ratio);
    //        int height = (int)(maxHeight * minRatio + maxHeight * (1.0f - minRatio) * ratio);

    //        switch (m_rndGen.Next(0, trainingSetHints.TargetImageVariability))
    //        {
    //            case 0:
    //                m_target = new GameObject(GameObjectType.None, @"White10x10.png", m_agent.X, m_agent.Y, width, height);
    //                break;
    //            case 1:
    //            default:
    //                m_target = new GameObject(GameObjectType.None, @"WhiteCircle50x50.png", m_agent.X, m_agent.Y, width, height);
    //                break;

    //        }
    //        m_target.isBitmapAsMask = trainingSetHints.ColorPatterns;
    //        m_target.maskColor = m_w.BackgroundColor;
    //        LearningTaskHelpers.RandomizeColorWDiff(ref m_target.maskColor, 0.1f, m_rndGen);

    //        m_w.AddGameObject(m_target);

    //        // first implementation, random object position (left or right)
    //        float maxDistanceRatioX = m_rndGen.Next(0, (trainingSetHints.TargetDistanceLevels + 1)) / (float)trainingSetHints.TargetDistanceLevels;
    //        float maxDistanceRatioY = m_rndGen.Next(0, (trainingSetHints.TargetDistanceLevels + 1)) / (float)trainingSetHints.TargetDistanceLevels;
    //        int maxTargetDistanceX = (int)(maxDistanceRatioX * (m_w.POW_WIDTH - m_target.Width) / 2.0f);
    //        int maxTargetDistanceY = (int)(maxDistanceRatioY * (m_w.POW_HEIGHT - m_target.Height) / 2.0f);

    //        bool isLeft = m_rndGen.Next(0, 2) == 1;
    //        if (isLeft)
    //            maxTargetDistanceX = -maxTargetDistanceX;

    //        bool isUp = m_rndGen.Next(0, 2) == 1;
    //        if (isUp)
    //            maxTargetDistanceY = -maxTargetDistanceY;

    //        m_target.X += maxTargetDistanceX;
    //        m_target.Y += maxTargetDistanceY;

    //        // center:
    //        m_target.X -= (m_target.Width) / 2;
    //        m_target.Y -= (m_target.Height) / 2;
    //    }
    //}
}
