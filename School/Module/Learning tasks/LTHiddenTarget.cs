using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    // A target (dummy target or reward target)
    public class HiddenTarget : GameObject
    {
        // Random numbers
        protected static Random m_rand = new Random();

        // Instantiates and installs the target
        public HiddenTarget(
            ManInWorld world,
            int imageIndex,
            float targetSizeStandardDeviation) :
            base(GameObjectType.None, GetShapeAddr(imageIndex), 0, 0)
        {
            int size1D = GetSize(targetSizeStandardDeviation);
            Point location = PickLocation(world, new Size(size1D, size1D));
            X = location.X;
            Y = location.Y;

            Height = Width = size1D;

            world.AddGameObject(this);
        }

        private static int GetSize(float targetSizeStandardDeviation)
        {
            const int DEFAULT_SIZE = 32;
            int size = DEFAULT_SIZE;

            if (targetSizeStandardDeviation != 0)
            {
                float scalingFactor = (float)Math.Pow(2, targetSizeStandardDeviation * LearningTaskHelpers.GetRandomGaussian(m_rand));
                size = (int)(scalingFactor * size);
            }

            return size;
        }

        // Gets the image corresponding to the shape index.
        public static string GetShapeAddr(int shapeIndex)
        {
            switch (shapeIndex)
            {
                case 0:
                    return @"WhiteL50x50.png";
                case 1:
                    return @"WhiteT50x50.png";
                case 2:
                    return @"WhiteCircle50x50.png";
                case 3:
                    return @"WhiteMountains50x50.png";
            }
            throw new ArgumentException("Unknown shape");
        }

        // Determine the placement of the target
        public static Point PickLocation(ManInWorld world, Size size)
        {
            // TODO Currently, degrees of freedom is not taken into account
            // And distance to target is the same throughout the learning task
            return world.RandomPositionInsidePowNonCovering(m_rand, size);

        }
    }

    /// <summary>
    /// In this learning task, the agent must learn to choose one of multiple targets without prior information as to which one to choose.
    /// </summary>
    public class LTHiddenTarget : AbstractLearningTask<RoguelikeWorld>
    {
        private static readonly TSHintAttribute NUMBER_OF_FALSE_TARGETS = new TSHintAttribute("Number of false targets", "", typeof(int), 1, 3);

        // Random numbers
        protected static Random m_rand = new Random();

        // The agent must learn to go to this target
        protected HiddenTarget rewardTarget;

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
                { TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 0 },
                { NUMBER_OF_FALSE_TARGETS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, .3f);
            TSProgression.Add(TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, .5f);
            TSProgression.Add(NUMBER_OF_FALSE_TARGETS, 2);
            TSProgression.Add(NUMBER_OF_FALSE_TARGETS, 3);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);
        }

        protected override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateAgent();

            int numberOfTargets = (int)TSHints[NUMBER_OF_FALSE_TARGETS] + 1;
            int rewardTargetIndex = m_rand.Next(numberOfTargets);
            for (int imageIndex = 0; imageIndex < numberOfTargets; imageIndex++)
            {
                HiddenTarget aTarget = CreateTarget(imageIndex);
                if (rewardTargetIndex == imageIndex)
                {
                    rewardTarget = aTarget;
                }
            }

            stepsSincePresented = 0;
            initialDistance = WrappedWorld.Agent.DistanceTo(rewardTarget);
        }

        protected HiddenTarget CreateTarget(int imageIndex)
        {
            return new HiddenTarget(
                WrappedWorld,
                imageIndex,
                TSHints[TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION]);
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            const float REQUIRED_PROXIMITY_TO_TARGET = 15;
            const int MAX_STEPS_TO_DISTANCE_RATIO = 3;

            bool didSucceed = WrappedWorld.Agent.DistanceTo(rewardTarget) < REQUIRED_PROXIMITY_TO_TARGET;
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
