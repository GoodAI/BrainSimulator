using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Pong with bricks" learning task</summary>
    /// <description>
    /// Ability info: Ability to play pong with bricks.
    /// The agent is presented the pong game, the expectancy is that the agent completes the game by passing all the required levels
    /// </description>
    public class LTPongWithBricks : AbstractLearningTask<PongAdapterWorld>
    {
        public LTPongWithBricks() : base(null) { }
        float ballHitSum;
        float ballMissSum;

        public readonly TSHintAttribute MAX_MISSES_ALLOWED = new TSHintAttribute("Maximum number of ball misses allowed before the training unit is declared failed", "", typeof(int), 0, 1);
        public readonly TSHintAttribute BALL_HITS_NEEDED = new TSHintAttribute("Ball hits needed in order to pass the training unit", "", typeof(int), 0, 1);

        public LTPongWithBricks(SchoolWorld w)
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

            WrappedWorld.UpdateTask.BOUNCE_BALL = 1.0f; // Set the default reward that is given upon hitting the ball to 1.0f (World's default value is 0.1f)
            WrappedWorld.UpdateTask.RandomBallDir = true;
            WrappedWorld.BricksEnabled = true;

            TSProgression.Add(TSHints.Clone());


            // TODO: modify Difficulty according also to current pong level
            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 1 },
                    { BALL_HITS_NEEDED, 1 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 6 },
                    { BALL_HITS_NEEDED, 10 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 6 },
                    { BALL_HITS_NEEDED, 20 }
            });


            TSProgression.Add(
                new TrainingSetHints {
                    { MAX_MISSES_ALLOWED, 5 },
                    { BALL_HITS_NEEDED, 30 }
            });


        }

        protected override void PresentNewTrainingUnit()
        {
            ballHitSum = 0f;
            ballMissSum = 0f;
            WrappedWorld.UpdateTask.ResetGame();

            WrappedWorld.UpdateTask.SetLevel();
        }

        public override void UpdateState()
        {
            base.UpdateState();

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
