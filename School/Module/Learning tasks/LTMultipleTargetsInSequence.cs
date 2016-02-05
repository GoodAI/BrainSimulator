using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Multiple targets in a sequence" learning task</summary>
    /// <description>
    /// Ability description: The agent learns to navigate to multiple targets in a defined sequence. One target type is always the first target, another target type is always the second and so on.
    /// </description>
    public class LTMultipleTargetsSequence : AbstractLearningTask<RoguelikeWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected GameObject m_agent;
        protected int m_stepsSincePresented = 0;

        public readonly TSHintAttribute SEQUENCE_LENGTH = new TSHintAttribute("Sequence length", "", TypeCode.Single, 0, 1);
        public readonly TSHintAttribute TIMESTEPS_LIMIT = new TSHintAttribute("Timesteps limit", "", TypeCode.Single, 0, 1);

        List<GameObject> GameObjectReferences = new List<GameObject>();     // Create a vector of references to GameObjects, we will need the index for the sequence order
        public int currentIndex = 0;                                        // Represents current index of the sequence


        public LTMultipleTargetsSequence() { }

        public LTMultipleTargetsSequence(SchoolWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;               // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS] = 1;
            TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS] = 10000;

            TSHints.Add(SEQUENCE_LENGTH, 1);
            TSHints.Add(TIMESTEPS_LIMIT, 2000);                             // Training unit fails if the TIMESTEP_LIMIT is reached, this is used to avoid that the agent stands without moving

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(SEQUENCE_LENGTH, 2);
            TSProgression.Add(SEQUENCE_LENGTH, 3);
            TSProgression.Add(SEQUENCE_LENGTH, 4);
            

            SetHints(TSHints);
        }


        public override void UpdateState()                                  // UpdateState calls base's equivalent and then its own additional functions
        {
            base.UpdateState();

            UpdateSequenceState();
        }


        protected override void PresentNewTrainingUnit()
        {
            CreateAgent();
            CreateTargets();

            m_stepsSincePresented = 0;
        }

        protected void CreateAgent()
        {
            //WrappedWorld.CreateAgent(@"Agent.png", WrappedWorld.FOW_WIDTH / 2, WrappedWorld.FOW_HEIGHT / 2);

            WrappedWorld.CreateAgent();
            m_agent = WrappedWorld.Agent;

            m_agent.X = WrappedWorld.FOW_WIDTH / 2;
            m_agent.Y = WrappedWorld.FOW_HEIGHT / 2;
        }

        public void CreateTargets()                                                                                     // Create a number of targets and fill the list(GameObjectReferences) of references for the sequence indexing
        {
            Point PositionFree = new Point();
            currentIndex = 0;
            GameObjectReferences.Clear();

            for (int n = 0; n < ((int)TSHints[SEQUENCE_LENGTH]); n++)                                                   // Generate a number of targets corresponding to the length of the sequence
            {
                m_target = new GameObject(GameObjectType.None, GetNumberTargetImage(n), 0, 0);
                WrappedWorld.AddGameObject(m_target);

                m_target.Width = 20;
                m_target.Height = 30;

                GameObjectReferences.Add(m_target);                                                                     // Add the current GameObject to the vector of GameObject references 



                PositionFree = WrappedWorld.RandomPositionInsidePowNonCovering(m_rndGen, m_target.GetGeometry().Size);  // Generate a random point where the corresponding GameObject doesn't cover any other GameObject

                m_target.X = PositionFree.X;                                                                            // Position target in the free position
                m_target.Y = PositionFree.Y;
            }

            //MyLog.DEBUG.WriteLine("Number of references created: " + GameObjectReferences.Count);
        }


        // Given an integer parameter, returns ther relevant path to its texture
        protected virtual string GetNumberTargetImage(int number)
        {
            switch (number)
            {
                case 0:
                    return "Number0_30x50.png";
                case 1:
                    return "Number1_30x50.png";
                case 2:
                    return "Number2_30x50.png";
                case 3:
                    return "Number3_30x50.png";
                case 4:
                    return "Number4_30x50.png";
                case 5:
                    return "Number5_30x50.png";
                default:
                    return "Number0_30x50.png";
            }
        }


        public void UpdateSequenceState()
        {
            float dist = m_agent.DistanceTo(GameObjectReferences[currentIndex]);        // Check if the agent reached the GameObject corresponding to the current index of the sequence, this is done by reference
            if (dist < 15)
            {
                // TODO: check if it's more appropriate having a delete function in ManInWorld
                WrappedWorld.gameObjects.Remove(GameObjectReferences[currentIndex]);    // If agent reached the right object (the one corresponding to the current index sequence), delete corresponding object
                currentIndex++;                                                         // And now change the index (the index will denote a reference to the GameObject that needs to be reached next)
            }
        }


        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;

            // Training unit is completed if the last index of the sequence is reached AND the last GameObject is reached
            if ((currentIndex == ((int)TSHints[SEQUENCE_LENGTH]) - 1) && (m_agent.DistanceTo(GameObjectReferences[currentIndex]) < 15))
            {
                wasUnitSuccessful = true;
                return true;
            }

            //MyLog.DEBUG.WriteLine("StepsSincePresented: " + m_stepsSincePresented);
            //MyLog.DEBUG.WriteLine("LIMIT in TIMESTEPS: " + TSHints[TIMESTEPS_LIMIT]);

            if (m_stepsSincePresented >= (int)TSHints[TIMESTEPS_LIMIT])                 // If the limit of timesteps is reached, declare the current training unit failed     
            {
                wasUnitSuccessful = false;
                return true;
            }

            wasUnitSuccessful = false;
            return false;
        }



    }
}
