using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <summary>
    /// In this learning task, the agent must learn to choose one of multiple targets without prior information as to which one to choose.
    /// </summary>
    public class LTHiddenTarget : AbstractLearningTask<RoguelikeWorld>
    {
        private static readonly TSHintAttribute NUMBER_OF_FALSE_TARGETS = new TSHintAttribute("Number of false targets", "", typeof(int), 1, 3);

        // Random numbers
        protected static Random m_rand = new Random();

        // The agent must learn to go to this target
        protected Shape m_rewardTarget;

        // Tracks the number of steps since TU presentation to determine if the TU has failed
        protected int stepsSincePresented = 0;

        // Tracks the initial agent-target distance to determine if the TU has failed
        protected float initialDistance = 0;

        public LTHiddenTarget() : base(null) { }

        // Construct the learning task
        public LTHiddenTarget(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { NUMBER_OF_FALSE_TARGETS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1f);
            TSProgression.Add(NUMBER_OF_FALSE_TARGETS, 2);
            TSProgression.Add(NUMBER_OF_FALSE_TARGETS, 3);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);
            TSProgression.Add(TSHintAttributes.IMAGE_TEXTURE_BACKGROUND, 1);
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateAgent();

            int numberOfTargets = (int)TSHints[NUMBER_OF_FALSE_TARGETS] + 1;
            int rewardTargetIndex = m_rand.Next(numberOfTargets);
            for (int imageIndex = 0; imageIndex < numberOfTargets; imageIndex++)
            {
                Shape aTarget = CreateTarget(imageIndex);
                if (rewardTargetIndex == imageIndex)
                {
                    m_rewardTarget = aTarget;
                }
            }

            stepsSincePresented = 0;
            initialDistance = WrappedWorld.Agent.DistanceTo(m_rewardTarget);
        }

        protected Shape CreateTarget(int imageIndex)
        {

            Size size = new Size(15,15);
            if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] >= 1f)
            {
                int a = m_rand.Next(10, 25);
                size = new Size(a, a);
            }

            // positions are reduced to keep all objects in POW

            Rectangle r = WrappedWorld.GetPowGeometry();
            r.Location = new Point(r.X + r.Width / 4, r.Y + r.Height / 4);
            r.Size = new Size(r.Width / 2, r.Height / 2);

            Point p = WrappedWorld.RandomPositionInsideRectangleNonCovering(m_rand, size, r, 4);

            Shape s = (Shape)WrappedWorld.CreateShape(p, (Shape.Shapes)imageIndex, Color.White, size, GameObjectType.NonColliding);

            return s;
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            const float REQUIRED_PROXIMITY_TO_TARGET = 7;
            const int MAX_STEPS_TO_DISTANCE_RATIO = 3;

            bool didSucceed = WrappedWorld.Agent.DistanceTo(m_rewardTarget) < REQUIRED_PROXIMITY_TO_TARGET;
            if (didSucceed)
            {
                wasUnitSuccessful = true;
                return true;
            }

            // We assume this method is called once per simulation step
            // There should be a better way to notify the LT of a new simulation step
            bool didFail = ++stepsSincePresented > MAX_STEPS_TO_DISTANCE_RATIO * initialDistance;
            if (didFail)
            {
                wasUnitSuccessful = false;
                return true;
            }

            return false;
        }
    }
}
