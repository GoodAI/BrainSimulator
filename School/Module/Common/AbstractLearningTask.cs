using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Worlds;
using System;
using System.Linq;

namespace GoodAI.Modules.School.Common
{
    public interface ILearningTask
    {
        bool HasPresentedFirstUnit { get; set; }
        bool IsAbilityLearned { get; }
        bool DidAbilityFail { get; }
        bool IsTrainingUnitCompleted { get; set; }

        bool IsInitialized { get; set; }

        TrainingSetHints TSHints { get; set; }
        TrainingSetProgression TSProgression { get; set; }
        int NumberOfLevels { get; set; }
        int CurrentLevel { get; set; }
        int NumberOfSuccessesRequired { get; }

        void UpdateState();
        bool HandlePresentNewTrainingUnit();

        SchoolWorld SchoolWorld { get; set; }
        Type RequiredWorld { get; set; }
        string Description { get; set; }

        void StartLearningTask();

        bool Solve(bool successfully);
    }

    /// <summary>
    /// Abstract base class for AI School exercises.
    /// </summary>
    public abstract class AbstractLearningTask<WrappedWorldClass> : ILearningTask where WrappedWorldClass : MyWorld, IWorldAdapter
    {
        public AbilityNameEnum AbilityName { get; set; }
        public AbilityNameEnum[] RequiredAbilities { get; set; }

        // True if the world is reset before presenting a new unit; true by default
        protected bool DoResetWorldBeforeTrainingUnit { get; set; }

        // Tracks whether the training unit has run to completion
        public bool IsTrainingUnitCompleted { get; set; }

        // True if the initialization step has been run
        public bool IsInitialized { get; set; }

        // True if the first training unit has been presented
        public bool HasPresentedFirstUnit { get; set; }

        // The number of consecutive examples in the training set classified correctly
        protected int CurrentNumberOfSuccesses { get; set; }

        // The number of training units so far in the training set
        protected int CurrentNumberOfAttempts { get; set; }

        // Number of consecutive successful classifications required to complete a level
        public virtual int NumberOfSuccessesRequired
        {
            get { return 20; }  // default value
        }

        // Current level of difficulty
        public int CurrentLevel { get; set; }

        // Parameters for training / testing
        public TrainingSetHints TSHints { get; set; }

        // List of parameters changing at each step of learning progression
        public TrainingSetProgression TSProgression { get; set; }

        // Number of levels of increasing difficulty
        private int m_numberOfLevels = -1;
        public int NumberOfLevels
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
        public SchoolWorld SchoolWorld { get; set; }

        // The wrapped world (e.g., a RoguelikeWorld or a TetrisWorld) that defines the agent's environment.
        // The learning task can access it to control the environment.
        public WrappedWorldClass WrappedWorld
        {
            get
            {
                return SchoolWorld.CurrentWorld as WrappedWorldClass;
            }
        }

        public Type RequiredWorld { get; set; }

        public string Description { get; set; }

        // Implement to manage challenge levels and training set hints
        protected virtual void UpdateLevel()
        {
            // We assume that levels are traversed sequentially.
            // Random access of levels would require a change of
            // implementation.

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSProgression[CurrentLevel]);

            MyLog.Writer.WriteLine(MyLogLevel.INFO,
                "Next level settings: \n" +
                TSHints.ToString()
                );
        }

        protected virtual void SetHints(TrainingSetHints trainingSetHints)
        {
            SchoolWorld.SetHints(trainingSetHints);
        }

        public virtual bool DidAbilityFail
        {
            get
            {
                if (TSHints.ContainsKey(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS))
                    return CurrentNumberOfAttempts >= TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS];
                else
                    return false;
            }
        }

        public virtual bool IsAbilityLearned
        {
            get { return CurrentLevel == NumberOfLevels - 1 && IsTrainingSetCompleted(); }
        }

        public AbstractLearningTask() : this(null) { }

        public AbstractLearningTask(SchoolWorld schoolWorld)
        {
            SchoolWorld = schoolWorld;
            TSHints = new TrainingSetHints();
            TSProgression = new TrainingSetProgression();
            CurrentNumberOfAttempts = CurrentNumberOfSuccesses = 0;
            CurrentLevel = 0;
            IsTrainingUnitCompleted = false;
            HasPresentedFirstUnit = false;
            DoResetWorldBeforeTrainingUnit = true;
            IsInitialized = false;
        }

        public virtual void UpdateState()
        {
            if (HasPresentedFirstUnit)
            {
                bool wasUnitSuccessful = false;

                if (SchoolWorld.IsEmulatingUnitCompletion())
                {
                    IsTrainingUnitCompleted = SchoolWorld.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
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

                    MyLog.Writer.WriteLine(MyLogLevel.INFO,
                        this.GetType().ToString().Split(new[] { '.' }).Last() +
                        " unit ends with result: " +
                        (wasUnitSuccessful ? "succes" : "fail") +
                        ". " +
                        CurrentNumberOfSuccesses + " succesful attepts in row, " +
                        NumberOfSuccessesRequired + " required."
                        );
                }
            }
        }

        // Implement to define the training set completion criterion
        protected virtual bool IsTrainingSetCompleted()
        {
            return CurrentNumberOfSuccesses >= NumberOfSuccessesRequired;
        }

        public virtual bool HandlePresentNewTrainingUnit()
        {
            bool didIncreaseLevel = false;

            if (IsTrainingSetCompleted())
            {
                CurrentLevel++;
                CurrentNumberOfAttempts = 0;
                CurrentNumberOfSuccesses = 0;
                UpdateLevel();
                didIncreaseLevel = true;
            }

            IsTrainingUnitCompleted = false;
            if (DoResetWorldBeforeTrainingUnit)
            {
                SchoolWorld.ClearWorld(TSHints);
            }
            PresentNewTrainingUnit();
            HasPresentedFirstUnit = true;

            return didIncreaseLevel;
        }

        protected abstract void PresentNewTrainingUnit();
        protected abstract bool DidTrainingUnitComplete(ref bool wasUnitSuccessful);

        public virtual void StartLearningTask()
        {
            SetHints(TSHints);
            IsInitialized = true;
        }

        public virtual bool Solve(bool successfully)
        {
            throw new NotImplementedException();
        }
    }
}
