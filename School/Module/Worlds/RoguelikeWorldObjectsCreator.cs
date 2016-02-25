using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.Worlds
{
    public partial class RoguelikeWorld
    {

        public override MovableGameObject CreateAgent()
        {
            RogueAgent agent = CreateAgent(GetInitPosition()) as RogueAgent;
            return agent;
        }

        public override MovableGameObject CreateAgent(PointF p, float size = 1.0f)
        {
            RogueAgent agent = new RogueAgent(p, size);
            AddGameObject(agent);
            Agent = agent;
            return agent;
        }

        public override MovableGameObject CreateNonVisibleAgent()
        {
            RogueAgent agent = new RogueAgent(GetInitPosition(), null);
            AddGameObject(agent);
            Agent = agent;
            return agent;
        }

        public MovableGameObject CreateTeacher(PointF p, List<RogueTeacher.Actions> actions)
        {
            Teacher = new RogueTeacher(p, actions);
            AddGameObject(Teacher);
            return Teacher;
        }

        public override GameObject CreateWall(PointF p, float size = 1.0f)
        {
            RogueWall w = new RogueWall(p, size);
            AddGameObject(w);
            return w;
        }

        public override GameObject CreateTarget(PointF p, float size = 1.0f)
        {
            RogueTarget t = new RogueTarget(p, size);
            AddGameObject(t);
            return t;
        }

        public override MovableGameObject CreateMovableTarget(PointF p, float size = 1.0f)
        {
            RogueMovableTarget mt = new RogueMovableTarget(p, size);
            AddGameObject(mt);
            return mt;
        }

        public override GameObject CreateDoor(PointF p, bool isClosed = true, float size = 1.0f)
        {
            RogueDoor rd = new RogueDoor(p, isClosed, size);
            AddGameObject(rd);
            return rd;
        }

        public override GameObject CreateLever(PointF p, bool isOn = false, float size = 1.0f)
        {
            RogueLever rl = new RogueLever(p, scale: size);
            AddGameObject(rl);
            return rl;
        }

        public override GameObject CreateLever(PointF p, ISwitchable obj, bool isOn = false, float size = 1.0f)
        {
            RogueLever rl = new RogueLever(p, obj, isOn, size);
            AddGameObject(rl);
            return rl;
        }

        public override GameObject CreateRogueKiller(PointF p, float size = 1.0f)
        {
            RogueKiller rk = new RogueKiller(p, size);
            AddGameObject(rk);
            return rk;
        }

        public override MovableGameObject CreateRogueMovableKiller(PointF p, float size = 1.0f)
        {
            RogueMovableKiller rmk = new RogueMovableKiller(p, size);
            AddGameObject(rmk);
            return rmk;
        }
    }
}
