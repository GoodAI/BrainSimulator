using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTConditionalTarget : AbstractLearningTask<ManInWorld>
    {
        public const string CONDITION_SALIENCE = "Condition salience";

        public LTConditionalTarget(ManInWorld w) : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 0 },
                { TSHintAttributes.TARGET_IMAGE_VARIABILITY, 1 },
                { TSHintAttributes.NOISE, 0 },
                { CONDITION_SALIENCE, 1 },
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.GIVE_PARTIAL_REWARDS, 1 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { TSHintAttributes.MAX_TARGET_DISTANCE, .3f }
            };

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);
            TSProgression.Add(TSHintAttributes.MAX_TARGET_DISTANCE, -1);
            TSProgression.Add(TSHintAttributes.NOISE, 1);
            TSProgression.Add(TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 1);
            TSProgression.Add(TSHintAttributes.TARGET_SIZE_STANDARD_DEVIATION, 1.5f);
            TSProgression.Add(TSHintAttributes.TARGET_IMAGE_VARIABILITY, 2);
            TSProgression.Add(TSHintAttributes.TARGET_IMAGE_VARIABILITY, 3);
            TSProgression.Add(TSHintAttributes.GIVE_PARTIAL_REWARDS, 0);
            TSProgression.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 100);

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
