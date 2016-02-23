using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;

namespace GoodAI.Modules.School.Common
{
    public enum TrainingResult
    {
        // the descriptions are used in GUI to show progress of an LT
        [Description("")]
        None,
        [Description("Running")]
        TUInProgress,
        [Description("Running")]
        FinishedTU,
        [Description("Failed")]
        FailedLT,
        [Description("Running")]
        FinishedLevel, // Implies FinishedTU
        [Description("Success")]
        FinishedLT, // Implies FinishedLevel
    }

    public interface ILearningTask
    {
        // GUI
        bool IsInitialized { get; set; }

        TrainingSetHints TSHints { get; set; }
        TrainingSetProgression TSProgression { get; set; }
        int NumberOfLevels { get; set; }
        int CurrentLevel { get; set;  }
        int NumberOfSuccessesRequired { get; }
        float Progress { get; }
        float Reward { get; set; }
        int CurrentNumberOfAttempts { get; set; }

        void ExecuteStep();
        TrainingResult EvaluateStep();
        void PresentNewTrainingUnitCommon();
        void IncreaseLevel();

        SchoolWorld SchoolWorld { get; set; }
        Type RequiredWorldType { get; set; }
        string GetTypeName();

        void Init();

        bool Solve(bool successfully);
    }

    /// <summary>
    /// Abstract base class for AI School exercises.
    /// </summary>
    public abstract class AbstractLearningTask<WrappedWorldClass> : ILearningTask where WrappedWorldClass : MyWorld, IWorldAdapter
    {
        public AbilityNameEnum AbilityName { get; set; }
        public AbilityNameEnum[] RequiredAbilities { get; set; }

        // The number of consecutive examples in the training set classified correctly
        public int CurrentNumberOfSuccesses { get; set; }

        // The number of training units so far in the training set
        public int CurrentNumberOfAttempts { get; set; }

        // Number of consecutive successful classifications required to complete a level
        public virtual int NumberOfSuccessesRequired
        {
            get { return 20; }  // default value
        }

        // True if the initialization step has been run
        public bool IsInitialized { get; set; }

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
                return m_numberOfLevels;
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

        public Type RequiredWorldType { get; set; }

        public virtual float Progress
        {
            get { return 100 * CurrentLevel / NumberOfLevels; }
        }

        // Implement to manage challenge levels and training set hints
        public virtual void IncreaseLevel()
        {
            CurrentLevel++;
            if (CurrentLevel >= NumberOfLevels)
            {
                // this case happens when the task is finished
                if(CurrentLevel > NumberOfLevels)
                {
                    // this case should not happen
                    Debug.Assert(false);
                }
                CurrentLevel = NumberOfLevels;
                return;
            }
            // We assume that levels are traversed sequentially.
            // Random access of levels would require a change of
            // implementation.
            CurrentNumberOfAttempts = 0;
            CurrentNumberOfSuccesses = 0;

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSHints);

            MyLog.Writer.WriteLine(MyLogLevel.INFO,
                "Next level settings: \n" +
                TSHints);
        }

        protected virtual void SetHints(TrainingSetHints trainingSetHints)
        {
            SchoolWorld.SetHints(trainingSetHints);
        }

        public AbstractLearningTask() : this(null) { }

        public AbstractLearningTask(SchoolWorld schoolWorld)
        {
            SchoolWorld = schoolWorld;
            TSHints = new TrainingSetHints();
            TSProgression = new TrainingSetProgression();
            CurrentNumberOfAttempts = CurrentNumberOfSuccesses = 0;
            CurrentLevel = 0;
            IsInitialized = false;
        }

        public virtual void ExecuteStep() { }

        public TrainingResult EvaluateStep()
        {
            // Check for unit completion
            bool wasUnitSuccessful = false;
            bool inProgress = (SchoolWorld.IsEmulatingUnitCompletion() && !SchoolWorld.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful))
                             || (!SchoolWorld.IsEmulatingUnitCompletion() && !DidTrainingUnitComplete(ref wasUnitSuccessful));

            if (inProgress)
            {
                // The unit is still in progress
                return TrainingResult.TUInProgress;
            }
            // otherwise the unit is over

            CurrentNumberOfAttempts++;

            // Check for task failure
            if (TSHints.ContainsKey(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS)
                && CurrentNumberOfAttempts >= TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS])
            {
                // Too many attempts
                return TrainingResult.FailedLT;
            }

            if (wasUnitSuccessful)
            {
                Reward = 1.0f;
                CurrentNumberOfSuccesses++;
            }
            else
            {
                CurrentNumberOfSuccesses = 0;
            }

            MyLog.Writer.WriteLine(
                MyLogLevel.INFO,
                GetTypeName() +
                " unit ends with result: " +
                (wasUnitSuccessful ? "success" : "fail") +
                ". " +
                CurrentNumberOfSuccesses + " successful attempts in row, " +
                NumberOfSuccessesRequired + " required.");


            // Check for level completion
            if (CurrentNumberOfSuccesses < NumberOfSuccessesRequired)
                // The level is still in progress
                return TrainingResult.FinishedTU;

            return TrainingResult.FinishedLevel;
        }

        public string GetTypeName()
        {
            return GetType().ToString().Split('.').Last();
        }

        public void PresentNewTrainingUnitCommon()
        {
            SchoolWorld.ClearWorld();
            SchoolWorld.SetHints(TSHints);
            Reward = 0.0f;

            PresentNewTrainingUnit();
        }

        public abstract void PresentNewTrainingUnit();
        protected abstract bool DidTrainingUnitComplete(ref bool wasUnitSuccessful);

        public void Init()
        {
            CurrentNumberOfAttempts = 0;
            CurrentNumberOfSuccesses = 0;
            CurrentLevel = 0;

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSHints);

            IsInitialized = true;
        }

        public virtual bool Solve(bool successfully)
        {
            throw new NotImplementedException();
        }

        public float Reward
        {
            get
            {
                return SchoolWorld.Reward;
            }
            set
            {
                SchoolWorld.Reward = value;
            }
        }
    }
}
