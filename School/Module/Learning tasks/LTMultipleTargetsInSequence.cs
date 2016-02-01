using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Multiple targets in a sequence" learning task</summary>
    /// <description>
    /// Ability Name: Multiple targets in a sequence
    /// </description>
    public class LTMultipleTargetsSequence : AbstractLearningTask<RoguelikeWorld>
    {

        protected int m_stepsSincePresented = 0;

        public LTMultipleTargetsSequence(RoguelikeWorld w)
            : base(w)
        {
        }



        protected override void PresentNewTrainingUnit()
        {
            m_stepsSincePresented = 0;
        }


        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;


            wasUnitSuccessful = false;
            return false;
        }


    }
}
