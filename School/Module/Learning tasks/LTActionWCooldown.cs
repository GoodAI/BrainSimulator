using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTActionWCooldown : AbstractLearningTask<ManInWorld>
    {
        private TSHintAttribute COOLDOWN = new TSHintAttribute("Cooldown value", "", typeof(float), 0, 1);
        private TSHintAttribute ACTION_LIMIT = new TSHintAttribute("Limit of number of steps waiting for action", "", typeof(float), 2, 10);

        private ManInWorld m_world { get; set; }
        private uint m_cooldownRemaining { get; set; }
        private Random m_rnd = new Random();
        //agent must perform action within this time limit, otherwise it is considered as a fail
        private int m_actionReadyFor;

        public uint UnitSuccesses { get; set; }
        public uint UnitAttempts { get; set; }

        public LTActionWCooldown() : base(null) { }

        public LTActionWCooldown(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {COOLDOWN,  5},
                {ACTION_LIMIT,  10},
                {TSHintAttributes.RANDOMNESS_LEVEL, 0 },
                {TSHintAttributes.REQUIRED_UNIT_SUCCESSES, 10 },
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            
        }

        protected override void PresentNewTrainingUnit()
        {
            ResetCooldown();
            UnitSuccesses = 0;
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (UnitSuccesses >= TSHints[TSHintAttributes.REQUIRED_UNIT_SUCCESSES])
                return wasUnitSuccessful = true;
            if (UnitAttempts >= TSHints[TSHintAttributes.MAX_UNIT_ATTEMPTS])
            {
                wasUnitSuccessful = false;
                return true;
            }
            return wasUnitSuccessful = false;
        }

        protected override void UpdateLevel()
        {
            var tsp = TSProgression[0];
            tsp[COOLDOWN] = 5 * (CurrentLevel + 1);
            tsp[TSHintAttributes.RANDOMNESS_LEVEL] = CurrentLevel < 20 ? 0 : CurrentLevel - 19;
            TSHints.Set(tsp);
        }

        private void ResetCooldown()
        {
            m_cooldownRemaining = (uint)TSHints[COOLDOWN];
            int randomness = (int)TSHints[TSHintAttributes.RANDOMNESS_LEVEL];
            m_cooldownRemaining += (uint)m_rnd.Next(-randomness * 3, randomness * 3);
            m_actionReadyFor = 0;
        }

        public override void UpdateState()
        {
            base.UpdateState();

            int step = (int)m_world.ExecutionBlock.Children[0].SimulationStep;
            ManInWorld manInWorld = m_world as ManInWorld;
            if (m_world == null)
                return;

            manInWorld.Controls.SafeCopyToHost();

            if (manInWorld.Controls.Host[0] > 0.5)
            {
                UnitAttempts++;

                if (m_cooldownRemaining <= 0)
                {
                    if (m_actionReadyFor < TSHints[ACTION_LIMIT])
                    {
                        manInWorld.Reward.Host[0] = 1;
                        UnitSuccesses++;
                    }
                    else
                        manInWorld.Reward.Host[0] = 0;
                    ResetCooldown();
                }
                else
                    manInWorld.Reward.Host[0] = -1;
            }
            else
                manInWorld.Reward.Host[0] = 0;

            manInWorld.Reward.SafeCopyToDevice();

            m_cooldownRemaining--;
            if (m_cooldownRemaining < 0)
                m_actionReadyFor++;
        }
    }
}
