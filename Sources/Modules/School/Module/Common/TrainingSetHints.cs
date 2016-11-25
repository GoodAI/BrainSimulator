using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.School.Common
{

    public class TSHintAttribute
    {
        public string Name { get; private set; }
        public string Annotation { get; protected set; }
        public Type TypeOfValue { get; protected set; }
        public Tuple<float, float> Range { get; protected set; }

        public TSHintAttribute(string name, string annotation, Type valueType, float lowerBound, float upperbound)
        {
            Name = name;
            Annotation = annotation;
            TypeOfValue = valueType;
            Range = new Tuple<float, float>(lowerBound, upperbound);
        }

        public override bool Equals(object obj)
        {
            return this.Name == (obj as TSHintAttribute).Name;
        }

        public override int GetHashCode()
        {
            return Name.GetHashCode();
        }
    }

    // !!! refactored to be more implementation independent and more operator understandable
    public static class TSHintAttributes
    {
        // A value [0, 1] measuring the amount of noise in the image
        public static readonly TSHintAttribute IMAGE_NOISE = new TSHintAttribute(
            "Image noise",
            "Adding gaussian noise to agent's POW.",
            typeof(bool),
            0, 1);

        public static readonly TSHintAttribute IMAGE_NOISE_BLACK_AND_WHITE = new TSHintAttribute(
            "Image noise color",
            "Image noise is black and white if on",
            typeof(bool),
            0, 1);

        public static readonly TSHintAttribute IMAGE_TEXTURE_BACKGROUND = new TSHintAttribute(
            "Image background",
            "Adding texture to the background of the world.",
            typeof(bool),
            0, 1);

        // Initially set this to some large number (10 000). Do not change it later!
        public static readonly TSHintAttribute MAX_NUMBER_OF_ATTEMPTS = new TSHintAttribute(
            "Max number of attempts",
            "Maximum number of attempts for whole learning task. If threshold is reached, simulation stops.", // TODO
            typeof(int),
            0, 1);

        public static readonly TSHintAttribute IS_VARIABLE_POSITION = new TSHintAttribute(
            "Variable position of objects",
            "Indicates whether positions of objects are variable/random.",
            typeof(bool),
            0, 1);

        public static readonly TSHintAttribute IS_VARIABLE_SIZE = new TSHintAttribute(
            "Variable size of objects",
            "Indicates whether sizes of objects are variable/random.",
            typeof(bool),
            0, 1);

        public static readonly TSHintAttribute IS_VARIABLE_COLOR = new TSHintAttribute(
            "Variable color", 
            "Indicates whether colours of objects are variable/random.",
            typeof(bool),
            0, 1);

        public static readonly TSHintAttribute IS_VARIABLE_ROTATION = new TSHintAttribute(
            "Variable rotation",
            "Indicates whether rotations of objects are variable/random.",
            typeof(bool),
            0, 1);

        // Randomness of whole learning task
        // For hints which are not covered by IS_VARIABLE_X
        /*public static readonly TSHintAttribute RANDOMNESS_LEVEL = new TSHintAttribute(
            "Randomness level",
            "",
            typeof(bool),
            0, 1);*/

        // True if the agent is rewarded at each step for approaching the target
        public static readonly TSHintAttribute GIVE_PARTIAL_REWARDS = new TSHintAttribute(
            "Give partial rewards",
            "Partial rewards are submitting during run of training unit.",
            typeof(bool),
            0, 1);

        // The number of dimensions in which the agent can move (1 or 2)
        public static readonly TSHintAttribute DEGREES_OF_FREEDOM = new TSHintAttribute(
            "Degrees of freedom",
            "The number of dimensions in which the agent can move (1 or 2).",
            typeof(int),
            1, 2);

        // Standard deviation of target size scaling (0 for fixed size)
        // The target size is obtained by multiplying with 2^s,
        // where s is normally distributed with mean = 0 and
        // standard deviation as specified
        //
        // replaced with VARIABLE_SIZE (binary)
        /*public static readonly TSHintAttribute DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION = new TSHintAttribute(
            "DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION",
            "",
            typeof(float),
            0, 1);*/

        // Estimate of cardinality the set of all visible objects
        public static readonly TSHintAttribute NUMBER_OF_DIFFERENT_OBJECTS = new TSHintAttribute(
            "Number of different objects.",
            "Estimate of cardinality the set of all visible objects.",
            typeof(int),
            0, 1);

        public static readonly TSHintAttribute NUMBER_OBJECTS = new TSHintAttribute(
            "Number of objects",
            "Total number of objects on the scene.",
            typeof(int),
            0, 1);

        // Max target distance as a multiple [0, 1] of the world size.
        // If non-negative, the distance between agent and target is uniformly distributed
        // between zero and this value. Otherwise, the target location is chosen randomly
        // from the entire image.
        //
        // use RandomPositionInsidePOW() instead;
        /*public static readonly TSHintAttribute DEPRECATED_MAX_TARGET_DISTANCE = new TSHintAttribute("DEPRECATED_MAX_TARGET_DISTANCE", "", typeof(bool), 0, 1);
        public static readonly TSHintAttribute DEPRECATED_TARGET_MAX_SIZE = new TSHintAttribute("DEPRECATED_TARGET_MAX_SIZE", "", typeof(float), 0, 1);
        public static readonly TSHintAttribute DEPRECATED_TARGET_MIN_SIZE = new TSHintAttribute("DEPRECATED_TARGET_MIN_SIZE", "", typeof(float), 0, 1);*/

        // is used in one task only
        // public static readonly TSHintAttribute DEPRECATED_COOLDOWN = new TSHintAttribute("DEPRECATED_COOLDOWN", "", typeof(float), 0, 1);

        //public static readonly TSHintAttribute REQUIRED_UNIT_SUCCESSES = new TSHintAttribute("REQUIRED_UNIT_SUCCESSES", "", typeof(int), 0, 1);
        //public static readonly TSHintAttribute MAX_UNIT_ATTEMPTS = new TSHintAttribute("MAX_UNIT_ATTEMPTS", "", typeof(int), 0, 1);
    }

    /// <summary>
    /// Training set hints are concrete parameters that
    /// determine the difficulty of a task, such as presence of noise or the
    /// number of attempts that the agent gets to complete the task.
    /// The values of the hints depend on the ChallengeLevel. Hints are world
    /// independent, and can sometimes be used/interpreted differently depending
    /// on the world.
    /// </summary>
    public class TrainingSetHints : Dictionary<TSHintAttribute, float>
    {
        public TrainingSetHints()
            : base()
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

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("{\n");

            foreach (var item in this)
            {
                sb.Append(item.Key.Name).Append(": ").Append(item.Value).Append("\n");
            }

            sb.Append("}");
            return sb.ToString();
        }
    }

    // Records the parameters to change at each step in the
    // learning challenge progression.
    public class TrainingSetProgression : List<TrainingSetHints>
    {

        public void Add(TSHintAttribute attribute, float value)
        {
            TrainingSetHints tsHints = new TrainingSetHints();
            tsHints.Add(attribute, value);
            Add(tsHints);
        }
    }
}
