using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Obstacles" learning task</summary>
    /// <description>
    /// Ability Name: Navigate to the target efficiently by avoiding obstacles
    /// </description>
    public class LTObstacles : AbstractLearningTask<RoguelikeWorld>
    {
        protected Random m_rndGen = new Random();
        protected GameObject m_target;
        protected GameObject m_agent;
        protected int m_stepsSincePresented = 0;
        protected float m_initialDistance = 0;

        private readonly TSHintAttribute OBSTACLES_LEVEL = new TSHintAttribute("Obstacles level", "", TypeCode.Single, 0, 1);   //check needed;
        private readonly TSHintAttribute TIMESTEPS_LIMIT = new TSHintAttribute("Timesteps limit", "", TypeCode.Single, 0, 1);   //check needed;

        public LTObstacles() { }

        public LTObstacles(SchoolWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;               // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS] = 1;
            TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS] = 10000;

            TSHints.Add(OBSTACLES_LEVEL, 1);
            TSHints.Add(TIMESTEPS_LIMIT, 800);
            
            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(OBSTACLES_LEVEL, 2);
            TSProgression.Add(OBSTACLES_LEVEL, 3);
            TSProgression.Add(OBSTACLES_LEVEL, 4);
            
            SetHints(TSHints);
        }

        protected override void PresentNewTrainingUnit()
        {
            GenerateObstacles();

            m_stepsSincePresented = 0;
            //m_initialDistance = m_agent.DistanceTo(m_target);
        }

        //Agent and target are generated in the same function, because the correspondent positions will depend on the wall positions
        public void GenerateObstacles()
        {
            
            int level = (int)TSHints[OBSTACLES_LEVEL];                                          // Update value of current level

            WrappedWorld.CreateAgent();                                                         // Generate agent
            m_agent = WrappedWorld.Agent;

            // Generate target
            m_target = new GameObject(GameObjectType.None, "White10x10.png", 0, 0);             // TODO: Change temporary "White10x10.png" with appropriate texture
            WrappedWorld.AddGameObject(m_target);

            RoguelikeWorld world = WrappedWorld as RoguelikeWorld;                              // Reference to World
            Grid g = world.GetGrid();                                                           // Get grid


            if (level == 1)
            {
                int widthOfRectangle = 10;
                int heightOfRectangle = 5;

                createWallRectangle(world, widthOfRectangle, heightOfRectangle);

                // Position agent
                m_agent.X = 50;
                m_agent.Y = 50;

                // Position target
                m_target.X = 280;
                m_target.Y = 120;
            }


            if (level == 2)                                                                     // Like level 1, but inverted
            {
                int widthOfRectangle = 10;
                int heightOfRectangle = 5;

                createWallRectangle(world, widthOfRectangle, heightOfRectangle);

                // Position agent
                m_agent.X = 280;
                m_agent.Y = 120;

                // Position target
                m_target.X = 50;
                m_target.Y = 50;
            }


            if (level == 3)
            {
                int widthOfRectangle = 20;
                int heightOfRectangle = 5;

                createWallRectangle(world, widthOfRectangle, heightOfRectangle);

                world.CreateWall(g.getPoint(14, 4));
                world.CreateWall(g.getPoint(14, 3));
                world.CreateWall(g.getPoint(14, 2));

                // Position agent
                m_agent.X = 80;
                m_agent.Y = 120;

                // Position target
                m_target.X = 550;
                m_target.Y = 110;
            }


            if (level == 4)
            {
                int widthOfRectangle = 20;
                int heightOfRectangle = 10;

                createWallRectangle(world, widthOfRectangle, heightOfRectangle);

                world.CreateWall(g.getPoint(14, 9));
                world.CreateWall(g.getPoint(14, 8));
                world.CreateWall(g.getPoint(14, 7));
                world.CreateWall(g.getPoint(14, 6));
                world.CreateWall(g.getPoint(14, 5));
                world.CreateWall(g.getPoint(14, 4));
                world.CreateWall(g.getPoint(14, 3));


                world.CreateWall(g.getPoint(8, 1));
                world.CreateWall(g.getPoint(8, 2));
                world.CreateWall(g.getPoint(8, 3));
                world.CreateWall(g.getPoint(8, 4));
                world.CreateWall(g.getPoint(8, 5));
                world.CreateWall(g.getPoint(8, 6));
                world.CreateWall(g.getPoint(8, 7));
                world.CreateWall(g.getPoint(8, 8));

                // Position agent
                m_agent.X = 80;
                m_agent.Y = 60;

                // Position target
                m_target.X = 550;
                m_target.Y = 250;
            }


        }


        public void createWallRectangle(RoguelikeWorld world, int widthOfRectangle, int heightOfRectangle)
        {
            Grid g = world.GetGrid();

            for (int k = 1; k < heightOfRectangle; k++)
            {
                world.CreateWall(g.getPoint(0, k));
                world.CreateWall(g.getPoint(widthOfRectangle, k));
            }
            for (int k = 1; k < widthOfRectangle; k++)
            {
                world.CreateWall(g.getPoint(k, 0));
                world.CreateWall(g.getPoint(k, heightOfRectangle));
            }

        }


        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;
            
            if (m_stepsSincePresented >= (int)TSHints[TIMESTEPS_LIMIT])
            {
                wasUnitSuccessful = false;
                return true;
            }

            float dist = m_agent.DistanceTo(m_target);
            if (dist < 15)
            {
                m_stepsSincePresented = 0;
                wasUnitSuccessful = true;
                return true;
            }

            wasUnitSuccessful = false;
            return false;
        }


    }


}
