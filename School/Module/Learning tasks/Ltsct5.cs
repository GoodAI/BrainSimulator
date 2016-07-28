using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("SC D1 LT5 - eat food")]
    public class Ltsct5 : Ltsct1
    {
        private readonly Random m_rndGen = new Random();

        public override string Path
        {
            get { return @"D:\summerCampSamples\SCT5\"; }
        }

        public Ltsct5() : this(null) { }

        public Ltsct5(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1},
                { TSHintAttributes.IMAGE_NOISE_BLACK_AND_WHITE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;

            return true;
        }

        
        protected override void CreateScene()
        {
            Actions = new AvatarsActions();

            if (m_rndGen.Next(3) <= 0)
            {
                SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

                PointF location = Positions.Center();

                if (LearningTaskHelpers.FlipCoin(m_rndGen))
                {
                    WrappedWorld.CreateRandomFood(location, size, m_rndGen);
                    Actions.Eat = true;
                }
                else
                {
                    WrappedWorld.CreateRandomStone(location, size, m_rndGen);
                }
            }

            Actions.WriteActions(StreamWriter);string joinedActions = Actions.ToString();MyLog.INFO.WriteLine(joinedActions);
        }
    }
}
