using GoodAI.Core.Nodes;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Drawing;


namespace GoodAI.Modules.School.Worlds{

    partial class PlumberWorld
    {

        public override MovableGameObject CreateAgent()
        {
            throw new NotImplementedException();
        }

        public override MovableGameObject CreateAgent(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override MovableGameObject CreateNonVisibleAgent()
        {
            MovableGameObject agent = CreateAgent(null, FOW_WIDTH / 2, FOW_HEIGHT / 2);
            Agent.IsAffectedByGravity = false;
            return agent;
        }

        public override GameObject CreateWall(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override GameObject CreateTarget(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override MovableGameObject CreateMovableTarget(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override GameObject CreateDoor(Point p, bool isClosed = true, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override GameObject CreateLever(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override GameObject CreateLever(Point p, ISwitchable obj, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override GameObject CreateRogueKiller(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override MovableGameObject CreateRogueMovableKiller(Point p, float size = 1.0f)
        {
            throw new NotImplementedException();
        }

        public override Grid GetGrid()
        {
            return new Grid(GetFowGeometry().Size, DEFAULT_GRID_SIZE);
        }
    }
}