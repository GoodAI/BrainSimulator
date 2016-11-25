using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;
using System.ComponentModel;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Pong without bricks" learning task</summary>
    /// <description>
    /// Ability info: Ability to play pong without bricks.
    /// The difficulty of the levels is denoted by the number of hits (paddle touching ball) needed in order to pass a training unit (BALL_HITS_NEEDED), and the maximum number of misses (ball reaching the bottom part of the screen untouched by the paddle, MAX_MISSES_ALLOWED)
    /// </description>
    [DisplayName("Pong without bricks")]
    public class LTPongWithoutBricks : AbstractLearningTask<PongAdapterWorld>
    {
        public LTPongWithoutBricks() : this(null) { }

        private float ballHitSum;
        private float ballMissSum;

        public readonly TSHintAttribute MAX_MISSES_ALLOWED = new TSHintAttribute("Maximum number of ball misses allowed before the training unit is declared failed", "", typeof(int), 0, 1);
        public readonly TSHintAttribute BALL_HITS_NEEDED = new TSHintAttribute("Ball hits needed in order to pass the training unit", "", typeof(int), 0, 1);

        public LTPongWithoutBricks(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSHints.Add(MAX_MISSES_ALLOWED, 1);
            TSHints.Add(BALL_HITS_NEEDED, 1);
            ballHitSum = 0f;
            ballHitSum = 0f;

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 1 },
                    { BALL_HITS_NEEDED, 1 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 10 },
                    { BALL_HITS_NEEDED, 6 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 10 },
                    { BALL_HITS_NEEDED, 10 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 10 },
                    { BALL_HITS_NEEDED, 20 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 10 },
                    { BALL_HITS_NEEDED, 30 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 10 },
                    { BALL_HITS_NEEDED, 40 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 1 },
                    { BALL_HITS_NEEDED, 10 }
            });
        }

        public override void PresentNewTrainingUnit()
        {
            ballHitSum = 0f;
            ballMissSum = 0f;
            WrappedWorld.UpdateTask.ResetGame();                            // Needed to restart from the default position when BALL_HITS_NEEDED is 1 and the training unit is already completed
            WrappedWorld.UpdateTask.BOUNCE_BALL = 1.0f;         // Set the default reward that is given upon hitting the ball to 1.0f (World's default value is 0.1f)
            WrappedWorld.UpdateTask.RandomBallDir = true;
        }

        public override void ExecuteStep()
        {
            ballHitSum += WrappedWorld.BinaryEvent.Host[0];
            ballMissSum += WrappedWorld.BinaryEvent.Host[2];
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            if (ballMissSum >= TSHints[MAX_MISSES_ALLOWED])
            {
                wasUnitSuccessful = false;
                return true;
            }

            if (ballHitSum >= TSHints[BALL_HITS_NEEDED])
            {
                wasUnitSuccessful = true;
                return true;
            }

            return false;
        }
    }
}
