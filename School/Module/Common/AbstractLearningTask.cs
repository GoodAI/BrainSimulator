using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Worlds;
using System;
using System.Linq;

namespace GoodAI.Modules.School.Common
{
    public interface ILearningTask
    {
        // GUI
        bool IsInitialized { get; set; }

        TrainingSetHints TSHints { get; set; }
        TrainingSetProgression TSProgression { get; set; }
        int NumberOfLevels { get; set; }
        int CurrentLevel { get; set; }
        int NumberOfSuccessesRequired { get; }

        void ExecuteStep();
        void EvaluateStep(out bool learningTaskFail);
        void PresentNewTrainingUnitCommon();

        SchoolWorld SchoolWorld { get; set; }
        Type RequiredWorld { get; set; }
        string Description { get; set; }
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
        protected int CurrentNumberOfSuccesses { get; set; }

        // The number of training units so far in the training set
        protected int CurrentNumberOfAttempts { get; set; }

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
        protected virtual bool IncreaseLevel()
        {
            // We assume that levels are traversed sequentially.
            // Random access of levels would require a change of
            // implementation.
            CurrentLevel++;
            if (CurrentLevel >= NumberOfLevels)
            {
                return false;
            }
            CurrentNumberOfAttempts = 0;
            CurrentNumberOfSuccesses = 0;

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSHints);

            return true;
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

        private bool DidLearingTaskFail()
        {
            if(TSHints.ContainsKey(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS))
            {
                return CurrentNumberOfAttempts >= TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS];
            }
            return false;
        }

        public virtual void ExecuteStep() { }

        public void EvaluateStep(out bool learningTaskFail)
        {
            learningTaskFail = false;
            bool wasUnitSuccessful = false;

            // evaluate whether is training unit complete
            bool trainingUnitIsComplete;
            if (SchoolWorld.IsEmulatingUnitCompletion())
            {
                // emulated validation
                trainingUnitIsComplete = SchoolWorld.EmulateIsTrainingUnitCompleted(out wasUnitSuccessful);
            }
            else
            {
                // real validation
                trainingUnitIsComplete = DidTrainingUnitComplete(ref wasUnitSuccessful);
            }

            // new training unit
            if (trainingUnitIsComplete)
            {
                CurrentNumberOfAttempts++;
                if (wasUnitSuccessful)
                {
                    CurrentNumberOfSuccesses++;
                }
                else
                {
                    CurrentNumberOfSuccesses = 0;
                }

                // if number of attempts reach its maximum, return with fail
                if (DidLearingTaskFail())
                {
                    learningTaskFail = true;
                    return;
                }

                MyLog.Writer.WriteLine(MyLogLevel.INFO,
                    GetTypeName() +
                    " unit ends with result: " +
                    (wasUnitSuccessful ? "success" : "fail") +
                    ". " +
                    CurrentNumberOfSuccesses + " successful attempts in row, " +
                    NumberOfSuccessesRequired + " required."
                    );
                // SchoolWorld.ClearWorld(TSHints);
                SchoolWorld.NotifyNewTrainingUnit();
            }
            // new level
            if (CurrentNumberOfSuccesses >= NumberOfSuccessesRequired)
            {
                bool didIncreaseLevel = IncreaseLevel();
                if(didIncreaseLevel)
                {
                    SchoolWorld.NotifyNewLevel();
                    // inform about new level
                    MyLog.Writer.WriteLine(MyLogLevel.INFO,
                        "Next level settings: \n" +
                        TSHints.ToString()
                        );
                }
                else // LT is over
                {
                    SchoolWorld.NotifyNewLearningTask();
                    return;
                }
            }
        }

        public string GetTypeName()
        {
            return this.GetType().ToString().Split(new[] { '.' }).Last();
        }

        public void PresentNewTrainingUnitCommon()
        {
            ManInWorld miw = WrappedWorld as ManInWorld;
            if (miw != null)
            {
                miw.IsImageNoise = TSHints[TSHintAttributes.IMAGE_NOISE] >= 1f;
            }
            
            PresentNewTrainingUnit();
        }

        public abstract void PresentNewTrainingUnit();
        protected abstract bool DidTrainingUnitComplete(ref bool wasUnitSuccessful);

        public void Init()
        {
            CurrentNumberOfAttempts = 0;
            CurrentLevel = 0;
            CurrentNumberOfSuccesses = 0;

            TSHints.Set(TSProgression[CurrentLevel]);
            SetHints(TSHints);

            SchoolWorld.NotifyNewTrainingUnit();
            SchoolWorld.NotifyNewLevel();

            IsInitialized = true;
        }

        public virtual bool Solve(bool successfully)
        {
            throw new NotImplementedException();
        }
    }
}
