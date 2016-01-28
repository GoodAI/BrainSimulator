using GoodAI.Core.Nodes;
using GoodAI.Modules.School.Worlds;
using System.Collections.Generic;

namespace GoodAI.Modules.School.Common
{
    public interface ILearningTask
    {
        bool HasPresentedFirstUnit { get; set; }
        bool IsAbilityLearned { get; }
        bool DidAbilityFail { get; }
        bool IsTrainingUnitCompleted { get; set; }

        void UpdateState();
        void HandlePresentNewTrainingUnit();
    }

    
    public interface IDeprecatedLearningTask
    {
        bool HasPresentedFirstUnit { get; set; }
        bool IsAbilityLearned { get; }
        bool DidAbilityFail { get; }
        bool IsTrainingUnitCompleted { get; set; }

        void UpdateState();
        void HandlePresentNewTrainingUnit(AbstractSchoolWorld w);    
    }

    /// <summary>
    /// Abstract base class for AI School exercises.
    /// </summary>
    public abstract class DeprecatedAbstractLearningTask<WorldClass> : IDeprecatedLearningTask where WorldClass : AbstractSchoolWorld
    {
        public LearningTaskNameEnum LearningTaskName { get; set; }
        public AbilityNameEnum AbilityName { get; set; }
        public AbilityNameEnum[] RequiredAbilities { get; set; }

        // True if the world is reset before presenting a new unit; true by default
        protected bool DoResetWorldBeforeTrainingUnit { get; set; }

        // Tracks whether the training unit has run to completion
        public bool IsTrainingUnitCompleted { get; set; }

        // True if the first training unit has been presented
        public bool HasPresentedFirstUnit { get; set; }

        // The number of consecutive examples in the training set classified correctly
        protected int CurrentNumberOfSuccesses { get; set; }

        // The number of training units so far in the training set
        protected int CurrentNumberOfAttempts { get; set; }

        // Number of consecutive successful classifications required to complete a level
        protected virtual int NumberOfSuccessesRequired
        {
            get { return 20; }  // default value
        }

        // Current level of difficulty
        protected int CurrentLevel { get; set; }

        // Parameters for training / testing
        protected TrainingSetHints TSHints { get; set; }

        // List of parameters changing at each step of learning progression
        protected TrainingSetProgression TSProgression { get; set; }

        // Number of levels of increasing difficulty
        private int m_numberOfLevels = -1;
        protected int NumberOfLevels
        {
            get
            {
                if (m_numberOfLevels < 0)
                {
                    return TSProgression.Count;
                }
                else
                {
                    return m_numberOfLevels;
                }
            }
            set
            {
                m_numberOfLevels = value;
            }
        }

        public WorldClass World { get; set; }

        // Implement to manage challenge levels and training set hints
        protected virtual void UpdateLevel()
        {
            // We assume that levels are traversed sequentially.
            // Random access of levels would require a change of
            // implementation.

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSProgression[CurrentLevel]);
        }

        protected virtual void SetHints(TrainingSetHints trainingSetHints)
        {
            World.SetHints(trainingSetHints);
        }

        public virtual bool DidAbilityFail
        {
            // TODO can we assume presence of this attribute???
            get { return CurrentNumberOfAttempts >= TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS]; }
        }

        public virtual bool IsAbilityLearned
        {
            get { return CurrentLevel == NumberOfLevels - 1 && IsTrainingSetCompleted(); }
        }

        public DeprecatedAbstractLearningTask() { }

        public DeprecatedAbstractLearningTask(WorldClass world)
        {
            World = world;
            TSHints = new TrainingSetHints();
            TSProgression = new TrainingSetProgression();
            CurrentNumberOfAttempts = CurrentNumberOfSuccesses = 0;
            CurrentLevel = 0;
            IsTrainingUnitCompleted = false;
            HasPresentedFirstUnit = false;
            DoResetWorldBeforeTrainingUnit = true;
        }

        public virtual void UpdateState()
        {
            if (HasPresentedFirstUnit)
            {
                bool wasUnitSuccessful = false;

                if (World.IsEmulatingUnitCompletion())
                {
                    IsTrainingUnitCompleted = World.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
                }
                else
                {
                    IsTrainingUnitCompleted = DidTrainingUnitComplete(ref wasUnitSuccessful);
                }

                if (IsTrainingUnitCompleted)
                {
                    if (wasUnitSuccessful)
                        CurrentNumberOfSuccesses++;
                    else
                        CurrentNumberOfSuccesses = 0;
                    CurrentNumberOfAttempts++;
                }
            }
        }

        // Implement to define the training set completion criterion
        protected virtual bool IsTrainingSetCompleted()
        {
            return CurrentNumberOfSuccesses >= NumberOfSuccessesRequired;
        }

        public virtual void HandlePresentNewTrainingUnit(AbstractSchoolWorld w)
        {
            if (IsTrainingSetCompleted())
            {
                CurrentLevel++;
                CurrentNumberOfAttempts = 0;
                CurrentNumberOfSuccesses = 0;
                UpdateLevel();
            }

            IsTrainingUnitCompleted = false;
            if (DoResetWorldBeforeTrainingUnit)
            {
                World.ClearWorld(TSHints);
            }
            PresentNewTrainingUnit();
            HasPresentedFirstUnit = true;
        }

        protected abstract void PresentNewTrainingUnit();
        protected abstract bool DidTrainingUnitComplete(ref bool wasUnitSuccessful);
    }

    /// <summary>
    /// Abstract base class for AI School exercises.
    /// </summary>
    public abstract class AbstractLearningTask<WrappedWorldClass> : ILearningTask where WrappedWorldClass : MyWorld, IWorldAdapter
    {
        public LearningTaskNameEnum LearningTaskName { get; set; }
        public AbilityNameEnum AbilityName { get; set; }
        public AbilityNameEnum[] RequiredAbilities { get; set; }

        // True if the world is reset before presenting a new unit; true by default
        protected bool DoResetWorldBeforeTrainingUnit { get; set; }

        // Tracks whether the training unit has run to completion
        public bool IsTrainingUnitCompleted { get; set; }

        // True if the first training unit has been presented
        public bool HasPresentedFirstUnit { get; set; }

        // The number of consecutive examples in the training set classified correctly
        protected int CurrentNumberOfSuccesses { get; set; }

        // The number of training units so far in the training set
        protected int CurrentNumberOfAttempts { get; set; }

        // Number of consecutive successful classifications required to complete a level
        protected virtual int NumberOfSuccessesRequired
        {
            get { return 20; }  // default value
        }

        // Current level of difficulty
        protected int CurrentLevel { get; set; }

        // Parameters for training / testing
        protected TrainingSetHints TSHints { get; set; }

        // List of parameters changing at each step of learning progression
        protected TrainingSetProgression TSProgression { get; set; }

        // Number of levels of increasing difficulty
        private int m_numberOfLevels = -1;
        protected int NumberOfLevels
        {
            get
            {
                if (m_numberOfLevels < 0)
                {
                    return TSProgression.Count;
                }
                else
                {
                    return m_numberOfLevels;
                }
            }
            set
            {
                m_numberOfLevels = value;
            }
        }

        // The world where the agent lives
        public SchoolWorld AdapterWorld { get; set; }

        // The wrapped world (e.g., a RoguelikeWorld or a TetrisWorld) that defines the agent's environment.
        // The learning task can access it to control the environment.
        public WrappedWorldClass WrappedWorld { get; set; }

        // Implement to manage challenge levels and training set hints
        protected virtual void UpdateLevel()
        {
            // We assume that levels are traversed sequentially.
            // Random access of levels would require a change of
            // implementation.

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSProgression[CurrentLevel]);
        }

        protected virtual void SetHints(TrainingSetHints trainingSetHints)
        {
            AdapterWorld.SetHints(trainingSetHints);
        }

        public virtual bool DidAbilityFail
        {
            // TODO can we assume presence of this attribute???
            get { return CurrentNumberOfAttempts >= TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS]; }
        }

        public virtual bool IsAbilityLearned
        {
            get { return CurrentLevel == NumberOfLevels - 1 && IsTrainingSetCompleted(); }
        }

        public AbstractLearningTask() { }

        public AbstractLearningTask(SchoolWorld adapterWorld)
        {
            AdapterWorld = adapterWorld;
            // I think we cannot avoid this cast? [SA]
            WrappedWorld = adapterWorld.CurrentWorld as WrappedWorldClass;
            TSHints = new TrainingSetHints();
            TSProgression = new TrainingSetProgression();
            CurrentNumberOfAttempts = CurrentNumberOfSuccesses = 0;
            CurrentLevel = 0;
            IsTrainingUnitCompleted = false;
            HasPresentedFirstUnit = false;
            DoResetWorldBeforeTrainingUnit = true;
        }

        public virtual void UpdateState()
        {
            if (HasPresentedFirstUnit)
            {
                bool wasUnitSuccessful = false;

                if (AdapterWorld.IsEmulatingUnitCompletion())
                {
                    IsTrainingUnitCompleted = AdapterWorld.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
                }
                else
                {
                    IsTrainingUnitCompleted = DidTrainingUnitComplete(ref wasUnitSuccessful);
                }

                if (IsTrainingUnitCompleted)
                {
                    if (wasUnitSuccessful)
                        CurrentNumberOfSuccesses++;
                    else
                        CurrentNumberOfSuccesses = 0;
                    CurrentNumberOfAttempts++;
                }
            }
        }

        // Implement to define the training set completion criterion
        protected virtual bool IsTrainingSetCompleted()
        {
            return CurrentNumberOfSuccesses >= NumberOfSuccessesRequired;
        }

        public virtual void HandlePresentNewTrainingUnit()
        {
            if (IsTrainingSetCompleted())
            {
                CurrentLevel++;
                CurrentNumberOfAttempts = 0;
                CurrentNumberOfSuccesses = 0;
                UpdateLevel();
            }

            IsTrainingUnitCompleted = false;
            if (DoResetWorldBeforeTrainingUnit)
            {
                AdapterWorld.ClearWorld(TSHints);
            }
            PresentNewTrainingUnit();
            HasPresentedFirstUnit = true;
        }

        protected abstract void PresentNewTrainingUnit();
        protected abstract bool DidTrainingUnitComplete(ref bool wasUnitSuccessful);
    }

}
