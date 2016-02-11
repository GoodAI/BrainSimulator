using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Worlds;
using System;
using System.Linq;

namespace GoodAI.Modules.School.Common
{
    public interface ILearningTask
    {
        TrainingSetHints TSHints { get; set; }
        TrainingSetProgression TSProgression { get; set; }
        int NumberOfLevels { get; set; }
        int CurrentLevel { get; set; }
        int NumberOfSuccessesRequired { get; }

        void ExecuteStep();
        void EvaluateStep();
        void PresentNewTrainingUnit();

        SchoolWorld SchoolWorld { get; set; }
        Type RequiredWorld { get; set; }
        string Description { get; set; }
        string GetTypeName();

        void Init();
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

            MyLog.Writer.WriteLine(MyLogLevel.INFO,
                "Next level settings: \n" +
                TSHints.ToString()
                );
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
        }

        public virtual void ExecuteStep() { }

        public void EvaluateStep()
        {
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

                if (wasUnitSuccessful)
                {
                    CurrentNumberOfSuccesses++;
                }
                else
                {
                    CurrentNumberOfSuccesses = 0;
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

        public abstract void PresentNewTrainingUnit();
        protected virtual void Update() { }
        protected abstract bool DidTrainingUnitComplete(ref bool wasUnitSuccessful);

        public void Init()
        {
            SetHints(TSHints);
            SchoolWorld.NotifyNewTrainingUnit();
            SchoolWorld.NotifyNewLevel();
        }
    }
}
