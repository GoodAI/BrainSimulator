using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;

namespace GoodAI.Modules.School.LearningTasks
{
    public class LTPongWithoutBricks : AbstractLearningTask<PongAdapterWorld>
    {
        public LTPongWithoutBricks() : base(null) { }
        float ballHitSum;
        float ballMissSum;

        public readonly TSHintAttribute MAX_MISSES_ALLOWED = new TSHintAttribute("Maximum number of ball misses allowed before the training unit is declared failed", "", typeof(int), 0, 1);
        public readonly TSHintAttribute BALL_HITS_NEEDED = new TSHintAttribute("Ball hits needed in order to pass the training unit", "", typeof(int), 0, 1);

        public LTPongWithoutBricks(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSHints.Add(MAX_MISSES_ALLOWED, 10);
            TSHints.Add(BALL_HITS_NEEDED, 4);
            ballHitSum = 0f;
            ballHitSum = 0f;

            WrappedWorld.UpdateTask.BOUNCE_BALL = 1.0f; // Set the default reward that is given upon hitting the ball to 1.0f (World's default value is 0.1f)

            TSProgression.Add(TSHints.Clone());

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


        }

        protected override void PresentNewTrainingUnit()
        {
            ballHitSum = 0f;
            ballMissSum = 0f;
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
