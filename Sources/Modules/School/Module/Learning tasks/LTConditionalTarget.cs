using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    // The condition is an arbitrary visual signal that tells the agent which target
    // to go to. It has two states (black and white).
    public class ConditionGameObject : Shape
    {
        // Size for salient condition
        private const int SALIENT_CONDITION_SIZE = 32;

        // Size for subtle condition
        private const int SUBTLE_CONDITION_SIZE = 16;

        // The condition has two states, black and white
        public bool IsWhite { get; set; }

        // Random numbers
        private static readonly Random m_rand = new Random();

        // Instantiates and installs the condition
        public ConditionGameObject(ManInWorld world, bool movingCondition, float salience) :
            base(Shapes.Square, PointF.Empty)
        {
            IsBitmapAsMask = true;
            IsWhite = LearningTaskHelpers.FlipCoin(m_rand);
            m_colorMask = IsWhite ? Color.White : Color.Black;

            float size1D = DetermineSize(salience);
            PointF location = PickLocation(world, movingCondition, new SizeF(size1D, size1D));
            Position = location;
            Size.Height = Size.Width = size1D;

            world.AddGameObject(this);
        }

        // Determine the size of the condition
        public static float DetermineSize(float salience)
        {
            // Currently we distinguish between two levels of
            // condition salience, corresponding to a huge condition
            // and a tiny one.
            return salience >= 1 ? SALIENT_CONDITION_SIZE : SUBTLE_CONDITION_SIZE;
        }

        // Determine the placement of the condition
        public static PointF PickLocation(ManInWorld world, bool movingCondition, SizeF size)
        {
            if (movingCondition)
            {
                return world.RandomPositionInsidePowNonCovering(m_rand, size, 1, 40);
            }

            const int FIXED_OFFSET = 10;
            RectangleF powRectangle = world.GetPowGeometry();
            return new PointF(powRectangle.X + FIXED_OFFSET, powRectangle.Y + FIXED_OFFSET);
        }

        public static float DistanceTo(PointF source, PointF target)
        {
            PointF vec = new PointF(target.X - source.X, target.Y - source.Y);
            return (float)Math.Sqrt(vec.X * vec.X + vec.Y * vec.Y);
        }
    }

    // A target (dummy target or reward target)
    public class ConditionalTarget : GameObject
    {
        // Random numbers
        private static readonly Random m_rand = new Random();

        // True if the target is indicated by the white condition state

        // Instantiates and installs the target
        public ConditionalTarget(
            ManInWorld world,
            bool isWhiteConditionTarget,
            float targetSizeStandardDeviation,
            int numberOfDifferentObjects /*,
            int degreesOfFreedom */) :
            base(PickShapePath(isWhiteConditionTarget, numberOfDifferentObjects))
        {
            int size1D = GetSize(targetSizeStandardDeviation);
            PointF location = PickLocation(world, new SizeF(size1D, size1D) /*, degreesOfFreedom */);
            Position = location;
            Size.Height = Size.Width = size1D;

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

        private static string PickShapePath(bool isWhiteConditionTarget, int numberOfDifferentObjects)
        {
            return GetShapeAddr(isWhiteConditionTarget, m_rand.Next(0, numberOfDifferentObjects));
        }

        // Gets the image corresponding to the shape index.
        // Each state of the condition (black, white) is associated with a different
        // pool of target images. Once the agent has learned to associate a target
        // image with a condition state, it will not have to unlearn the association.
        public static string GetShapeAddr(bool isWhiteConditionTarget, int shapeIndex)
        {
            if (isWhiteConditionTarget)
            {
                switch (shapeIndex)
                {
                    case 0:
                        return @"WhiteL50x50.png";
                    case 1:
                        return @"WhiteT50x50.png";
                    case 2:
                        return @"WhiteCircle50x50.png";
                }
            }
            else
            {
                switch (shapeIndex)
                {
                    case 0:
                        return @"WhiteMountains50x50.png";
                    case 1:
                        return @"WhitePentagon50x50.png";
                    case 2:
                        return @"WhiteRhombus50x50.png";
                }
            }

            throw new ArgumentException("Unknown shape");
        }

        // Determine the placement of the target
        public static PointF PickLocation(ManInWorld world, SizeF size /*, int degreesOfFreedom */)
        {
            // TODO Currently, degrees of freedom is not taken into account
            // And distance to target is the same throughout the learning task
            return world.RandomPositionInsidePowNonCovering(m_rand, size, 10, 22);
        }
    }

    /// <summary>
    /// In this learning task, the agent must learn to choose one of two targets depending on an arbitrary visual signal (the condition).
    /// </summary>
    [DisplayName("Reach cued target")]
    public class LTConditionalTarget : AbstractLearningTask<RoguelikeWorld>
    {
        // In the beginning, the condition doesn't move
        private static readonly TSHintAttribute MOVING_CONDITION = new TSHintAttribute("Moving condition", "If true, conditional square is placed randomly, otherwise in upper lef corner.", typeof(bool), 0, 1);

        // The condition can be more or less salient
        private static readonly TSHintAttribute CONDITION_SALIENCE = new TSHintAttribute("Condition square size", "", typeof(float), 1f, 0.5f);

        // Visual condition
        protected ConditionGameObject condition;

        // The agent gets no reward for going to this target
        protected ConditionalTarget dummyTarget;

        // The agent must learn to go to this target
        protected ConditionalTarget rewardTarget;

        // Tracks the number of steps since TU presentation to determine if the TU has failed
        protected int stepsSincePresented = 0;

        // Tracks the initial agent-target distance to determine if the TU has failed
        protected float initialDistance = 0;

        public LTConditionalTarget() : this(null) { }

        // Construct the learning task
        public LTConditionalTarget(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { MOVING_CONDITION, 0 },
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { CONDITION_SALIENCE, 1 },
                //{ TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                // Currently, target locations are always uniformly distributed inside POW
                //{ TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE, .3f }
                { TSHintAttributes.IMAGE_TEXTURE_BACKGROUND, 0 },
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(MOVING_CONDITION, 1);
            //TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);
            //TSProgression.Add(TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE, -1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3);
            TSProgression.Add(CONDITION_SALIENCE, .5f);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);
            TSProgression.Add(TSHintAttributes.IMAGE_TEXTURE_BACKGROUND, 1);
        }

        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateAgent();
            condition = new ConditionGameObject(WrappedWorld, TSHints[MOVING_CONDITION] == 1, TSHints[CONDITION_SALIENCE]);
            ConditionalTarget blackConditionTarget = CreateTarget(/* isWhiteConditionalTarget: */ false);
            ConditionalTarget whiteConditionTarget = CreateTarget(/* isWhiteConditionalTarget: */ true);
            dummyTarget = condition.IsWhite ? blackConditionTarget : whiteConditionTarget;
            rewardTarget = condition.IsWhite ? whiteConditionTarget : blackConditionTarget;
            stepsSincePresented = 0;
            initialDistance = WrappedWorld.Agent.DistanceTo(rewardTarget);
        }

        protected ConditionalTarget CreateTarget(bool isWhiteConditionTarget)
        {
            float stdDev = 0;
            if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] > 0)
            {
                stdDev = 0.5f;
            }
            return new ConditionalTarget(
                WrappedWorld,
                isWhiteConditionTarget,
                stdDev,
                (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS] /*,
                (int)TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] */);
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
