using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.School.Common
{

    // !!! refactored to be more implementation independent and more operator understandable
    public static class TSHintAttributes
    {
        // A value [0, 1] measuring the amount of noise in the image
        public const string IMAGE_NOISE = "Noise";

        public const string MAX_NUMBER_OF_ATTEMPTS = "Maximum number of attempts";

        public const string IS_VARIABLE_POSITION = "Variable position of objects";
        public const string IS_VARIABLE_SIZE = "Variable size of objects";
        public const string IS_VARIABLE_COLOR = "Variable color of objects";

        // Randomness of whole learning task
        // For hints which are not covered by IS_VARIABLE_X
        public const string RANDOMNESS_LEVEL = "Randomness";

        // True if the agent is rewarded at each step for approaching the target
        public const string GIVE_PARTIAL_REWARDS = "Give partial rewards";

        // The number of dimensions in which the agent can move (1 or 2)
        public const string DEGREES_OF_FREEDOM = "Degrees of freedom";

        // Standard deviation of target size scaling (0 for fixed size)
        // The target size is obtained by multiplying with 2^s,
        // where s is normally distributed with mean = 0 and
        // standard deviation as specified
        // 
        // replaced with VARIABLE_SIZE (biniry)
        public const string DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION = "Target size standard deviation";

        // Estimate of cardinality the set of all visible objects
        public const string NUMBER_OF_DIFFERENT_OBJECTS = "Target image variability";

        // Max target distance as a multiple [0, 1] of the world size.
        // If non-negative, the distance between agent and target is uniformly distributed
        // between zero and this value. Otherwise, the target location is chosen randomly
        // from the entire image.
        //
        // use RandomPositionInsidePOW() instead;
        public const string DEPRECATED_MAX_TARGET_DISTANCE = "Maximum target distance";

        public const string DEPRECATED_TARGET_MAX_SIZE = "Maximum target size";
        public const string DEPRECATED_TARGET_MIN_SIZE = "Minimum target size";

        public const string DEPRECATED_COOLDOWN = "Cooldown interval";

        public const string REQUIRED_UNIT_SUCCESSES = "Required unit success";
        public const string MAX_UNIT_ATTEMPTS = "Maximal unit attemts";
    }

    /// <summary>
    /// Training set hints are concrete parameters that
    /// determine the difficulty of a task, such as presence of noise or the
    /// number of attempts that the agent gets to complete the task.
    /// The values of the hints depend on the ChallengeLevel. Hints are world
    /// independent, and can sometimes be used/interpreted differently depending
    /// on the world.
    /// </summary>
    public class TrainingSetHints : Dictionary<string, float>
    {
        public TrainingSetHints() : base()
        {
        }

        public TrainingSetHints(TrainingSetHints other)
            : base(other)
        {
        }

        public TrainingSetHints Clone()
        {
            return new TrainingSetHints(this);
        }


        public void Set(TrainingSetHints trainingSetHints)
        {
            foreach (var tsHint in trainingSetHints)
            {
                this[tsHint.Key] = tsHint.Value;
            }
        }
    }

    // Records the parameters to change at each step in the 
    // learning challenge progression.
    public class TrainingSetProgression : List<TrainingSetHints>
    {

        public void Add(string attribute, float value)
        {
            TrainingSetHints tsHints = new TrainingSetHints();
            tsHints.Add(attribute, value);
            Add(tsHints);
        }
    }
}
