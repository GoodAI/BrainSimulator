using GoodAI.Modules.School.Common;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTConditionalTarget : AbstractLearningTask<ManInWorld>
    {
        private static readonly TSHintAttribute CONDITION_SALIENCE = new TSHintAttribute("Condition salience","",TypeCode.Single,0,1); //check needed;

        public LTConditionalTarget() { }

        public LTConditionalTarget(ManInWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 0 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { CONDITION_SALIENCE, 1 },
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE, .3f }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);
            TSProgression.Add(TSHintAttributes.DEPRECATED_MAX_TARGET_DISTANCE, -1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 1);
            TSProgression.Add(TSHintAttributes.DEPRECATED_TARGET_SIZE_STANDARD_DEVIATION, 1.5f);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);

            SetHints(TSHints);
            // TODO condition salience
        }


        protected override void PresentNewTrainingUnit()
        {

        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            return false;
        }

        //private IWorldAdapter MakeWorldAdapter(AbstractSchoolWorld w)
        //{
        //    if (w is RoguelikeWorld)
        //    {
        //        return new RoguelikeWorldWAConditionalTarget();
        //    }
        //    else if (w is PlumberWorld)
        //    {
        //        return new PlumberWorldWAConditionalTarget();
        //    }
        //    else
        //    {
        //        throw new System.ArgumentException("World must be RoguelikeWorld or PlumberWorld");
        //    }
        //}
    }

    // Abstract base class for world adapters

    //public abstract class AbstractWorldAdapter : IWorldAdapter
    //{
    //    protected Random m_rndGen = new Random();
    //    protected GameObject m_target;
    //    protected GameObject m_agent;
    //    protected abstract AbstractSchoolWorld World { get; }

    //    public void PresentNewTrainingUnit(AbstractSchoolWorld w, TrainingSetHints hints)
    //    {
    //        InstallWorld(w);
    //        CreateAgent(hints);
    //        CreateTarget(hints);
    //        SetUpEnvironment(hints);
    //    }

    //    protected abstract void InstallWorld(AbstractSchoolWorld w);

    //    protected void CreateAgent(TrainingSetHints trainingSetHints)
    //    {

    //    }

    //    protected void CreateTarget(TrainingSetHints trainingSetHints)
    //    {

    //    }

    //    protected void SetUpEnvironment(TrainingSetHints trainingSetHints)
    //    {

    //    }
    //}

    //public abstract class AbstractWAConditionalTarget : AbstractWorldAdapter
    //{
    //    // TODO
    //}

    //public class PlumberWorldWAConditionalTarget : AbstractWAConditionalTarget
    //{
    //    private PlumberWorld m_w;

    //    protected override AbstractSchoolWorld World
    //    {
    //        get
    //        {
    //            return m_w;
    //        }
    //    }


    //    // TODO
    //}

    //public class RoguelikeWorldWAConditionalTarget : AbstractWAConditionalTarget
    //{
    //    private RoguelikeWorld m_w;

    //    protected override AbstractSchoolWorld World
    //    {
    //        get
    //        {
    //            return m_w;
    //        }
    //    }

    //    // TODO
    //}
}
