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
        public TypeCode TypeOfValue { get; protected set; }
        public Tuple<float, float> Range { get; protected set; }

        public TSHintAttribute(string name, string annotation, TypeCode valueType, float lowerBound, float upperbound)
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
        private static TypeCode m_boolT = TypeCode.Boolean;
        private static TypeCode m_intT = TypeCode.Int32;
        private static TypeCode m_floatT = TypeCode.Decimal;

        // A value [0, 1] measuring the amount of noise in the image
        public static readonly TSHintAttribute IMAGE_NOISE = new TSHintAttribute(
            "Image noise", 
            "Adding noise to agent's POW. Color of each pixel is slightly randomly changed",
            m_boolT,
            0, 1);

        public static readonly TSHintAttribute MAX_NUMBER_OF_ATTEMPTS = new TSHintAttribute("MAX_NUMBER_OF_ATTEMPTS", "", m_boolT, 0, 1);

        public static readonly TSHintAttribute IS_VARIABLE_POSITION = new TSHintAttribute(
            "Variable position of objects",
            "",
            TypeCode.Boolean,
            0, 1);

        public static readonly TSHintAttribute IS_VARIABLE_SIZE = new TSHintAttribute("Variable size","",m_boolT,0,1);
        public static readonly TSHintAttribute IS_VARIABLE_COLOR = new TSHintAttribute("Variable color","",m_boolT,0,1);

        // Randomness of whole learning task
        // For hints which are not covered by IS_VARIABLE_X
        public static readonly TSHintAttribute RANDOMNESS_LEVEL = new TSHintAttribute("RANDOMNESS_LEVEL", "", m_boolT, 0, 1);

        // True if the agent is rewarded at each step for approaching the target
        public static readonly TSHintAttribute GIVE_PARTIAL_REWARDS = new TSHintAttribute("Give partial rewards","",m_boolT,0,1);

        // The number of dimensions in which the agent can move (1 or 2)
        public static readonly TSHintAttribute DEGREES_OF_FREEDOM = new TSHintAttribute("Degrees of freedom","",m_intT,1,2);

        // Standard deviation of target size scaling (0 for fixed size)
        // The target size is obtained by multiplying with 2^s,
        // where s is normally distributed with mean = 0 and
        // standard deviation as specified
        // 
        // replaced with VARIABLE_SIZE (binary)
        public static readonly TSHintAttribute DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION = new TSHintAttribute("DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION", "", m_floatT, 0, 1);

        // Estimate of cardinality the set of all visible objects
        public static readonly TSHintAttribute NUMBER_OF_DIFFERENT_OBJECTS = new TSHintAttribute("NUMBER_OF_DIFFERENT_OBJECTS", "", m_intT, 0, 1);

        // Max target distance as a multiple [0, 1] of the world size.
        // If non-negative, the distance between agent and target is uniformly distributed
        // between zero and this value. Otherwise, the target location is chosen randomly
        // from the entire image.
        //
        // use RandomPositionInsidePOW() instead;
        public static readonly TSHintAttribute DEPRECATED_MAX_TARGET_DISTANCE = new TSHintAttribute("DEPRECATED_MAX_TARGET_DISTANCE", "", m_boolT, 0, 1);

        public static readonly TSHintAttribute DEPRECATED_TARGET_MAX_SIZE = new TSHintAttribute("DEPRECATED_TARGET_MAX_SIZE", "", m_floatT, 0, 1);
        public static readonly TSHintAttribute DEPRECATED_TARGET_MIN_SIZE = new TSHintAttribute("DEPRECATED_TARGET_MIN_SIZE", "", m_floatT, 0, 1);

        // is used in one task only
        public static readonly TSHintAttribute DEPRECATED_COOLDOWN = new TSHintAttribute("DEPRECATED_COOLDOWN", "", m_floatT, 0, 1);

        public static readonly TSHintAttribute REQUIRED_UNIT_SUCCESSES = new TSHintAttribute("REQUIRED_UNIT_SUCCESSES", "", m_intT, 0, 1);
        public static readonly TSHintAttribute MAX_UNIT_ATTEMPTS = new TSHintAttribute("MAX_UNIT_ATTEMPTS", "", m_intT, 0, 1);
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

        public void Add(TSHintAttribute attribute, float value)
        {
            TrainingSetHints tsHints = new TrainingSetHints();
            tsHints.Add(attribute, value);
            Add(tsHints);
        }
    }
}
