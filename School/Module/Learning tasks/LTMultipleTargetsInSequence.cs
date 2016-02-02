using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

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
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected GameObject m_agent;
        protected int m_stepsSincePresented = 0;

        public readonly TSHintAttribute SEQUENCE_LENGTH = new TSHintAttribute("Sequence length", "", TypeCode.Single, 0, 1);       //check needed;
        public readonly TSHintAttribute TIMESTEPS_LIMIT = new TSHintAttribute("Timesteps limit", "", TypeCode.Single, 0, 1);      //check needed;

        public LTMultipleTargetsSequence() { }

        public LTMultipleTargetsSequence(SchoolWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;               // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS] = 1;
            TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS] = 10000;

            TSHints.Add(SEQUENCE_LENGTH, 2);
            TSHints.Add(TIMESTEPS_LIMIT, 2000);                             // Training unit fails if the TIMESTEP_LIMIT is reached, this is used to avoid that the agent stands without moving

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(TSHintAttributes.DEGREES_OF_FREEDOM, 2);

            SetHints(TSHints);
        }


        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();
            CreateTargets();

            m_stepsSincePresented = 0;
        }

        protected void CreateAgent()
        {
            WrappedWorld.CreateAgent();
            m_agent = WrappedWorld.Agent;
            m_agent.X = 30;
            m_agent.Y = 30;
        }

        public void CreateTargets()
        {
            Point p1 = new Point();

            Size test = new Size();
            test.Width = 40;
            test.Height = 40;

            //for (int n = 0; n < ((int)TSHints[SEQUENCE_LENGTH]); n++)                          // Generate a number of targets corresponding to the length of the sequence
            for (int k = 0; k < 3; k++)
            {
                //p1 = World.RandomPositionInsidePow(m_rndGen1, World.GetPowGeometry().Size);
                //p1 = World.RandomPositionInsideFow(m_rndGen, World.GetFowGeometry().Size);

                p1 = WrappedWorld.RandomPositionInsidePow(m_rndGen, test);

                m_target = new GameObject(GameObjectType.None, GetTargetImage(k), 0, 0);
                WrappedWorld.AddGameObject(m_target);

                m_target.X = p1.X;
                m_target.Y = p1.Y;

                //m_target.X = (50 * k);
                //m_target.Y = p1.Y;
            }
        }


        protected virtual string GetTargetImage(int number)                                     // TODO: return string/path corresponding to the number given as parameter
        {
            switch (number)
            {
                case 0:
                    //return "Block60x10.png"; // used to be GetDefaultTargetImage();
                    return "Target_TOP.png";
                case 1:
                    return "White10x10.png";
                case 2:
                default:
                    return "WhiteCircle50x50.png";
            }
        }


        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;

            //MyLog.DEBUG.WriteLine("StepsSincePresented: " + m_stepsSincePresented);
            //MyLog.DEBUG.WriteLine("LIMIT in TIMESTEPS: " + TSHints[TIMESTEPS_LIMIT]);

            if (m_stepsSincePresented >= (int)TSHints[TIMESTEPS_LIMIT])
            {
                wasUnitSuccessful = false;
                return true;
            }

            /*
            //TODO: Insert here code of distance to the target corresponding to the current index of the sequence
            float dist = m_agent.DistanceTo(m_target);
            if (dist < 15)
            {
                m_stepsSincePresented = 0;
                wasUnitSuccessful = true;
                return true;
            }
            */

            wasUnitSuccessful = false;
            return false;
        }



    }
}
