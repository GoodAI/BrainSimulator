using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("Debugging")]
    public class LTDebugging : AbstractLearningTask<ManInWorld>
    {
        private GameObject m_target;
        private MovableGameObject m_agent;
        private Random m_rndGen = new Random();

        public LTDebugging() : this(null) { }

        public LTDebugging(SchoolWorld w)
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
            if (WrappedWorld.GetType() == typeof(PlumberWorld))
            {
                m_agent = new MovableGameObject(@"Plumber24x28.png", new PointF(24, 28), type: GameObjectType.Agent);
                PlumberWorld world = WrappedWorld as PlumberWorld;
                m_target = new GameObject(@"Coin16x16.png", new PointF(200, 200), type: GameObjectType.NonColliding);
                world.AddGameObject(m_target);

                GameObject obj1 = new GameObject(@"Block60x10.png", new PointF(10, 260));
                GameObject obj2 = new GameObject(@"Block60x10.png", new PointF(100, 250));
                GameObject obj3 = new GameObject(@"Block5x120.png", new PointF(200, 100));
                GameObject obj4 = new GameObject(@"Block60x10.png", new PointF(300, 200));

                world.AddGameObject(obj1);
                world.AddGameObject(obj2);
                world.AddGameObject(obj3);
                world.AddGameObject(obj4);
            }
            else if (WrappedWorld.GetType() == typeof(RoguelikeWorld))
            {
                RoguelikeWorld world = WrappedWorld as RoguelikeWorld;
                world.DegreesOfFreedom = 2;

                // create agent
                m_agent = world.CreateAgent();

                // get grid
                Grid g = world.GetGrid();

                // place objects according to the grid
                world.CreateWall(g.GetPoint(12, 17));
                m_agent.Position = g.GetPoint(15, 16);
                world.CreateWall(g.GetPoint(15, 17));
                world.CreateWall(g.GetPoint(16, 17));
                world.CreateWall(g.GetPoint(17, 17));
                world.CreateWall(g.GetPoint(17, 18));

                // place new wall with random position and size (1 - 3 times larger than default)
                GameObject wall = world.CreateWall(g.GetPoint(0, 0), (float)(1 + m_rndGen.NextDouble() * 2));
                // GetRandomPositionInsidePow avoids covering agent
                PointF randPosition = world.RandomPositionInsideViewport(m_rndGen, wall.GetGeometry().Size);
                wall.Position = randPosition;

                // create target
                m_target = world.CreateTarget(g.GetPoint(18, 18));

                // create shape
                world.CreateShape(// position
                    // Type determine interactions
                    // uses texture shape as mask
                    Shape.Shapes.Star, Color.Cyan, g.GetPoint(11, 13), new SizeF(30, 30), 60, GameObjectType.Obstacle); // you can resize choosen shape

                RogueDoor door = (RogueDoor)world.CreateDoor(g.GetPoint(13, 17), size: 2f);
                world.CreateLever(g.GetPoint(13, 13), door);
            }
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // check if target was reached
            if (m_agent.ActualCollisions.Contains(m_target)) // Collision check
            {
                Core.Utils.MyLog.INFO.WriteLine("Succes. End of unit!");
                return wasUnitSuccessful = true;
            }
            return wasUnitSuccessful = false;
        }
    }
}
