using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT2 - 2 shapes")]
    public class Ltsct2 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public Ltsct2() : this(null) { }

        public Ltsct2(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        protected override void CreateScene()
        {
            if (m_rndGen.Next(ScConstants.numShapes * ScConstants.numShapes + 1) == 1) return; // no shape, no target

            int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);

            AddShape(randomLocationIdx);

            if (m_rndGen.Next(ScConstants.numShapes + 1) == 1) return; // one shape only

            int nextRandomLocationIdx = m_rndGen.Next(randomLocationIdx + 1, randomLocationIdx + ScConstants.numPositions);
            nextRandomLocationIdx %= ScConstants.numPositions;

            AddShape(nextRandomLocationIdx);
        }
    }
}
