using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT2 - 2 shapes")]
    public class Ltsct2 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\SCT2\"; }
        }

        public Ltsct2() : this(null) { }

        public Ltsct2(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1},
                { TSHintAttributes.IMAGE_NOISE_BLACK_AND_WHITE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        protected override void CreateScene()
        {
            AvatarsActions actions = new AvatarsActions();

            int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape(randomLocationIdx);
            }

            actions.Shapes[ShapeIndex] = true;

            int nextRandomLocationIdx = m_rndGen.Next(randomLocationIdx + 1, randomLocationIdx + ScConstants.numPositions);
            nextRandomLocationIdx %= ScConstants.numPositions;

            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
                AddShape(nextRandomLocationIdx);
            }

            actions.Shapes[ShapeIndex] = true;

            actions.WriteActions(StreamWriter);
        }
    }
}
