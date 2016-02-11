using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"LTObstacles with POW visible target" learning task</summary>
    /// <description>
    /// The class is derived from LTObstacles, and implements the level where the target is always visible from POW and randomness Check LTObstacles for further details
    /// </description>
    public class LTObstaclesTargetOnSight : LTObstacles                           // Deriving from LTObstacles
    {
        public LTObstaclesTargetOnSight() : base(null) { }

        public LTObstaclesTargetOnSight(SchoolWorld w)
            : base(w)
        {
            TSHints.Clear();
            TSProgression.Clear();

            TSHints.Add(TIMESTEPS_LIMIT, 200);
            TSHints.Add(TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000);
            TSHints.Add(TSHintAttributes.IMAGE_NOISE, 0);

            TSProgression.Add(TSHints.Clone());
            TSProgression.Add(TIMESTEPS_LIMIT, 100);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TIMESTEPS_LIMIT, 50);
        }

        public override void PresentNewTrainingUnit()
        {
            GenerateObstacles();

            m_stepsSincePresented = 0;
            //m_initialDistance = m_agent.DistanceTo(m_target);
        }

        public void GenerateObstacles()
        {
            WrappedWorld.CreateAgent();                                                         // Generate agent
            m_agent = WrappedWorld.Agent;

            m_target = WrappedWorld.CreateTarget(new Point(0, 0));
            WrappedWorld.AddGameObject(m_target);

            RoguelikeWorld world = WrappedWorld as RoguelikeWorld;                              // Reference to World
            Grid g = world.GetGrid();                                                           // Get grid

            TSHints[TIMESTEPS_LIMIT] = 200;

            int widthOfRectangle = 3;
            int heightOfRectangle = 2;

            createWallRectangleWithFillProbability(world, m_rndGen, 5, 5, widthOfRectangle, heightOfRectangle, 0.8f, 1.0f);

            // Position agent
            m_agent.X = 200;
            m_agent.Y = 193;


            // Position target
            m_target.X = m_rndGen.Next(160, 280);

            if (m_rndGen.Next(0, 2) == 0)
            {
                m_target.Y = 120;
            }
            else
            {
                m_target.Y = 270;
            }
            
        }
    }
}