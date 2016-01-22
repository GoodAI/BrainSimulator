using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;


namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Moving Target" learning task</summary>
    /// <description>
    /// Ability Name: Efficient navigation in a simple environment + moving target.
    /// In this Learning task there is an agent and a moving target, to pass each level the agent needs to reach the target a defined number of consecutive times,
    /// There is a limit on the number of timesteps that can be used to reach the target, If this limit is reached, the level fails and the agent starts from the same level but the number of consecutive times is reset.
    /// Different levels will create targets with different moving characteristics.
    /// The teacher declares the ability as learned when the agent reaches the last level.
    /// </description>
    public class LTMovingTarget : AbstractLearningTask<ManInWorld>
    {

        public uint UnitSuccesses { get; set; }
        public uint UnitAttempts { get; set; }

        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected GameObject m_agent;
        protected int m_stepsSincePresented = 0;
        protected float m_initialDistance = 0;

        public const string TARGET_VX = "Target velocity X";
        public const string TARGET_VY = "Target velocity Y";

        public LTMovingTarget(ManInWorld w): base(w)
        {
            TSHints = new TrainingSetHints {
                {TARGET_VX, 1},
                {TARGET_VY, 0},
                {TSHintAttributes.REQUIRED_UNIT_SUCCESSES, 5 },
                {TSHintAttributes.MAX_UNIT_ATTEMPTS, 350 },
            };
            
            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TARGET_VX, 2);
            TSProgression.Add(TARGET_VX, 3);
            TSProgression.Add(new TrainingSetHints { { TARGET_VX, 4 }, {TARGET_VY, 1 }});
            TSProgression.Add(TARGET_VX, 6);

            SetHints(TSHints);

            // should it be here?
            World.FOW_WIDTH = 400;
            World.FOW_HEIGHT = 300;
            World.VisualFOW.ColumnHint = 400;
            World.VisualFOW.Reallocate(World.FOW_WIDTH * World.FOW_HEIGHT * 3);
        }

        public override void UpdateState()
        {
            base.UpdateState();

            UnitAttempts++;                                                   //Increase the count of frames, it's used for deciding if a training unit failed
            if (UnitAttempts > TSHints[TSHintAttributes.MAX_UNIT_ATTEMPTS])                    // If the number of timesteps reached the limit, restart level from the beginning
            {
                MyLog.DEBUG.WriteLine("Failed, Restarting: ");
                UnitAttempts = 0;
                UnitSuccesses = 0;
            }
        }

        // Adaptor to PlumberWorld

        protected override void PresentNewTrainingUnit()
        {
            World.Cleanup();
            CreateAgent();
            CreateTarget();
            UnitAttempts = 0;
            UnitSuccesses = 0;
        }

        protected void CreateAgent()
        {
            World.CreateAgent(@"Plumber24x28.png", m_rndGen.Next(0, World.FOW_WIDTH - 100), m_rndGen.Next(0, World.FOW_HEIGHT - 100));                        // Create Agent in a random X,Y position
            m_agent = World.Agent;                                                                                                                        // Create reference to Agent
        }

        protected void CreateTarget()
        {
            MovableGameObject Target = new MovableGameObject(GameObjectType.Enemy, @"Coin16x16.png", m_rndGen.Next(0, World.FOW_WIDTH - 100), 200);   // Create Target in a random X position
            Target.vX = TSHints[TARGET_VX];                                                                                                                          // Initialise X velocity (It will start by moving on the right)
            Target.vY = TSHints[TARGET_VY];                                                                                                                         // Y velocity is 0 (Doesn't move Up/Down)
            Target.GameObjectStyle = GameObjectStyleType.Pinball;                                                                                   // Define Object type as Pinball, so it bounces
            Target.IsAffectedByGravity = false;
            World.AddGameObject(Target);
            m_target = Target;                                                                                                                      // Create reference to Target
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (UnitAttempts >= TSHints[TSHintAttributes.MAX_UNIT_ATTEMPTS])
            {
                wasUnitSuccessful = false;
                return true;
            }

            if (m_agent.DistanceTo(m_target) < 15f)     // TrainingUnit is completed when the Agent reaches the coin
            {
                World.Reward.Host[0] = 1f;                // Temporary way of using the reward for testing
                World.Reward.SafeCopyToDevice();

                UnitSuccesses++;
                if (UnitSuccesses >= TSHints[TSHintAttributes.REQUIRED_UNIT_SUCCESSES])
                {
                    return wasUnitSuccessful = true;
                }
            }

            World.Reward.Host[0] = 0f;                    // Temporary way of using the reward for testing
            World.Reward.SafeCopyToDevice();

            return false;
        }
    }


    
}

