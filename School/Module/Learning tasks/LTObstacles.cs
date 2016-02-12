using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.Drawing;

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

        protected MovableGameObject m_movingObstacle1;
        protected MovableGameObject m_movingObstacle2;

        public readonly TSHintAttribute OBSTACLES_LEVEL = new TSHintAttribute("Obstacles level", "", typeof(int), 0, 1);
        public readonly TSHintAttribute TIMESTEPS_LIMIT = new TSHintAttribute("Timesteps limit", "", typeof(int), 0, 1);

        public LTObstacles() : base(null) { }

        public LTObstacles(SchoolWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.IMAGE_NOISE] = 0;
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;               // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS] = 1;
            TSHints[TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS] = 10000;

            TSHints.Add(OBSTACLES_LEVEL, 1);
            TSHints.Add(TIMESTEPS_LIMIT, 800);

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(OBSTACLES_LEVEL, 2);
            TSProgression.Add(OBSTACLES_LEVEL, 3);
            TSProgression.Add(OBSTACLES_LEVEL, 4);

            TSProgression.Add(OBSTACLES_LEVEL, 5);
            TSProgression.Add(OBSTACLES_LEVEL, 6);
        }

        public override void PresentNewTrainingUnit()
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

            m_target = WrappedWorld.CreateTarget(new Point(0, 0));
            WrappedWorld.AddGameObject(m_target);

            RoguelikeWorld world = WrappedWorld as RoguelikeWorld;                              // Reference to World
            Grid g = world.GetGrid();                                                           // Get grid


            if (level == 1)
            {
                TSHints[TIMESTEPS_LIMIT] = 500;

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
                TSHints[TIMESTEPS_LIMIT] = 500;

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
                TSHints[TIMESTEPS_LIMIT] = 400;

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
                TSHints[TIMESTEPS_LIMIT] = 400;

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


            if (level == 5)
            {
                TSHints[TIMESTEPS_LIMIT] = 300;

                int widthOfRectangle = 13;
                int heightOfRectangle = 5;

                createWallRectangle(world, widthOfRectangle, heightOfRectangle);

                // Walls positioned randomly inside the wall rectangle
                createWallVerticalLine(world, 4, m_rndGen.Next(1, 3), 3);
                createWallVerticalLine(world, 6, m_rndGen.Next(1, 3), 3);
                createWallVerticalLine(world, 8, m_rndGen.Next(1, 3), 3);

                // Position agent
                m_agent.X = 50;
                m_agent.Y = 50;

                // Position target according to the wall rectangle that was generated with random size
                m_target.X = 350;
                //m_target.Y = 110;
                m_target.Y = m_rndGen.Next(70, 110);   // Randomize Y position of target
            }

            if (level == 6)
            {
                TSHints[TIMESTEPS_LIMIT] = 260;

                int widthOfRectangle = 13;
                int heightOfRectangle = 5;

                createWallRectangle(world, widthOfRectangle, heightOfRectangle);

                // Walls positioned randomly inside the wall rectangle
                createWallVerticalLine(world, 4, m_rndGen.Next(1, 3), 3);
                createWallVerticalLine(world, 6, m_rndGen.Next(1, 3), 3);
                createWallVerticalLine(world, 8, m_rndGen.Next(1, 3), 3);

                // Position agent
                m_agent.X = 50;
                m_agent.Y = 50;

                // Position target according to the wall rectangle that was generated with random size
                m_target.X = 350;
                //m_target.Y = 110;
                m_target.Y = m_rndGen.Next(70, 110);   // Randomize Y position of target
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



        /*
         * The function uses the grid to create a partial rectangle with settable size and position, the rectangle is made of walls and the probability of the corresponding blocks being filled can be set.
         * "wallAddProbability" denotes the probability of each block of the rectangle being filled: it ranges from 0 (empty rectangle, no rectangle) to 1 (full rectangle), 0.5f means the rectangle is filled using half of the blocks
         * The algorithm ensures there is always 1 hole, even if "wallAddProbability" is 1.0f
         */
        public void createWallRectangleWithFillProbability(RoguelikeWorld world, Random rndGen, int gridX, int gridY, int widthOfRectangle, int heightOfRectangle, float wallAddProbability, float sizeWall)
        {
            Grid g = world.GetGrid();
            bool firstHoleEncountered = false;

            for (int k = 1; k < heightOfRectangle; k++) // Fill Y axis
            {
                if (Convert.ToSingle(m_rndGen.NextDouble()) < wallAddProbability)
                {
                    world.CreateWall(g.getPoint(gridX, k + gridY), sizeWall);
                }
                else
                {
                    firstHoleEncountered = true;
                }
                if (Convert.ToSingle(m_rndGen.NextDouble()) < wallAddProbability)
                {
                    world.CreateWall(g.getPoint(widthOfRectangle + gridX, k + gridY), sizeWall);
                }
                else
                {
                    firstHoleEncountered = true;
                }
            }

            for (int k = 1; k < widthOfRectangle; k++)  // Fill X axis
            {
                if (Convert.ToSingle(m_rndGen.NextDouble()) < wallAddProbability)
                {
                    world.CreateWall(g.getPoint(gridX + k, gridY), sizeWall);
                }
                else
                {
                    firstHoleEncountered = true;
                }

                if (Convert.ToSingle(m_rndGen.NextDouble()) < wallAddProbability)
                {
                    if (k == (widthOfRectangle - 1) && firstHoleEncountered == false)
                    {
                        // This case will happen to avoid the rectangle being without holes
                    }
                    else
                    {
                        world.CreateWall(g.getPoint(gridX + k, heightOfRectangle + gridY), sizeWall);
                    }

                }
                else
                {
                    firstHoleEncountered = true;
                }
            }
        }


        public void createWallHorizontalLine(RoguelikeWorld world, int gridX, int gridY, int lengthOfLine)
        {
            Grid g = world.GetGrid();

            for (int k = 0; k < lengthOfLine; k++)
            {
                world.CreateWall(g.getPoint(gridX + k, gridY));
            }
        }

        public void createWallVerticalLine(RoguelikeWorld world, int gridX, int gridY, int lengthOfLine)
        {
            Grid g = world.GetGrid();

            for (int k = 0; k < lengthOfLine; k++)
            {
                world.CreateWall(g.getPoint(gridX, gridY + k));
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // MyLog.DEBUG.WriteLine("X, Y = " + m_agent.X + ", " + m_agent.Y);
            // expect this method to be called once per simulation step
            m_stepsSincePresented++;

            if (m_stepsSincePresented >= (int)TSHints[TIMESTEPS_LIMIT])
            {
                m_stepsSincePresented = 0;
                wasUnitSuccessful = false;
                return true;
            }

            float dist = m_agent.DistanceTo(m_target);
            if (dist < 3)
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





// Wall rectangle with random width and corresponding target positioning
/*
if (level == 1)
{
    //int widthOfRectangle = 10;
    int widthOfRectangle = m_rndGen.Next(20, 30);

    int heightOfRectangle = 5;


    createWallRectangle(world, widthOfRectangle, heightOfRectangle);

    /*
    // Create bouncing block
    m_movingObstacle1 = new MovableGameObject(GameObjectType.Obstacle, "Armor_Block.png", 30, 60);
    m_movingObstacle1.Width = 30;
    m_movingObstacle1.Height = 60;
    m_movingObstacle1.GameObjectStyle = GameObjectStyleType.Pinball;
    m_movingObstacle1.IsAffectedByGravity = false;
    m_movingObstacle1.X = 98;
    m_movingObstacle1.Y = 60;
    m_movingObstacle1.vY = 2;
    WrappedWorld.AddGameObject(m_movingObstacle1);


    // Create bouncing block
    m_movingObstacle2 = new MovableGameObject(GameObjectType.Obstacle, "Armor_Block.png", 30, 60);
    m_movingObstacle2.Width = 30;
    m_movingObstacle2.Height = 60;
    m_movingObstacle2.GameObjectStyle = GameObjectStyleType.Pinball;
    m_movingObstacle2.IsAffectedByGravity = false;
    m_movingObstacle2.X = 170;
    m_movingObstacle2.Y = 90;
    m_movingObstacle2.vY = -2;
    WrappedWorld.AddGameObject(m_movingObstacle2);


    // Position agent
    m_agent.X = 50;
    m_agent.Y = 50;

    // Position target according to the wall rectangle that was generated with random size
    m_target.X = (widthOfRectangle * 30) - 30 ;
    m_target.Y = 120;
}
*/