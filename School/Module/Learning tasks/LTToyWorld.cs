using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.School.Worlds;

namespace GoodAI.School.Learning_tasks
{
    [DisplayName("Shows ToyWorld")]
    public class LTToyWorld : AbstractLearningTask<ToyWorldAdapterWorld>
    {
        public LTToyWorld() : this(null) { }

        public LTToyWorld(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                {TSHintAttributes.IMAGE_NOISE, 0},
                {TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000}
            };

            TSProgression.Add(TSHints.Clone());
        }

        public override void PresentNewTrainingUnit()
        {
            //throw new NotImplementedException();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;
            return false;
            //throw new NotImplementedException();
        }
    }
}
