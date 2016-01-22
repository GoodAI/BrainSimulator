using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.Worlds
{
    public class RoguelikeWorld : ManInWorld, IMyCustomTaskFactory
    {
        public static Size DEFAULT_GRID_SIZE = new Size(32, 32);

        protected override string TEXTURE_DIR { get { return @"res\RoguelikeWorld"; } }

        [MyTaskInfo(OneShot = true, Order = 90)]
        public class InitTask : InitSchoolTask
        {
            public override void Init(int nGPU)
            {
                base.Init(nGPU);
            }

            public override void Execute()
            {
                base.Execute();
            }
        }

        public override void Validate(Core.Utils.MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Controls != null, this, "Controls must not be null");
            if (Controls != null)
                validator.AssertError(Controls.Count >= 4, this, "Size of Controls must be 4 or more");
        }

        public virtual void CreateTasks()
        {
            InitSchool = new InitTask();
            GetInputTask = new GetRLInputTask();
            UpdateWorldTask = new UpdateRLTask();
        }

        public Point GetInitPosition()
        {
            if(Agent == null){
                return new Point(
                    FOW_WIDTH / 2 - RogueAgent.GetDefaultSize().Width / 2,
                    FOW_HEIGHT / 2 - RogueAgent.GetDefaultSize().Height / 2);
            }
            return new Point(FOW_WIDTH / 2 - Agent.Width / 2, FOW_HEIGHT / 2 - Agent.Height / 2);
        }

        public RogueAgent CreateAgent()
        {
            RogueAgent agent = CreateAgent(GetInitPosition());
            return agent;
        }

        public RogueAgent CreateAgent(Point p, float size = 1.0f)
        {
            RogueAgent agent = new RogueAgent(p, size);
            AddGameObject(agent);
            Agent = agent;
            return agent;
        }

        public RogueAgent CreateNonVisibleAgent()
        {
            RogueAgent agent = new RogueAgent(GetInitPosition(), null);
            AddGameObject(agent);
            Agent = agent;
            return agent;
        }

        public RogueTeacher CreateTeacher(Point p, List<RogueTeacher.Actions> actions)
        {
            Teacher = new RogueTeacher(p, actions);
            AddGameObject(Teacher);
            return Teacher as RogueTeacher;
        }

        public RogueWall CreateWall(Point p, float size = 1.0f)
        {
            RogueWall w = new RogueWall(p, size);
            AddGameObject(w);
            return w;
        }

        public RogueTarget CreateTarget(Point p, float size = 1.0f)
        {
            RogueTarget t = new RogueTarget(p, size);
            AddGameObject(t);
            return t;
        }

        public RogueMovableTarget CreateMovableTarget(Point p, float size = 1.0f)
        {
            RogueMovableTarget mt = new RogueMovableTarget(p, size);
            AddGameObject(mt);
            return mt;
        }

        public RogueDoor CreateDoor(Point p, bool isClosed = true, float size = 1.0f)
        {
            RogueDoor rd = new RogueDoor(p, isClosed : isClosed ,size : size);
            AddGameObject(rd);
            return rd;
        }

        public RogueLever CreateLever(Point p, float size = 1.0f)
        {
            RogueLever rl = new RogueLever(p, size : size);
            AddGameObject(rl);
            return rl;
        }

        public RogueLever CreateLever(Point p, ISwitchable obj, float size = 1.0f)
        {
            RogueLever rl = new RogueLever(p, obj, size);
            AddGameObject(rl);
            return rl;
        }

        public RogueKiller CreateRogueKiller(Point p, float size = 1.0f)
        {
            RogueKiller rk = new RogueKiller(p, size);
            AddGameObject(rk);
            return rk;
        }

        public RogueMovableKiller CreateRogueMovableKiller(Point p, float size = 1.0f)
        {
            RogueMovableKiller rmk = new RogueMovableKiller(p, size);
            AddGameObject(rmk);
            return rmk;
        }

        public Grid GetGrid()
        {
            return new Grid(GetFowGeometry().Size, DEFAULT_GRID_SIZE);
        }

        public class GetRLInputTask : InputTask
        {
            public override void Init(int nGPU)
            {
                base.Init(nGPU);
            }

            public override void Execute()
            {
                base.Execute();
            }
        }

        public class UpdateRLTask : UpdateTask
        {
            public override void Init(int nGPU)
            {
                base.Init(nGPU);
            }

            public override void HandleCollisions()
            {
                base.HandleCollisions();
            }

            public override void Execute()
            {
                base.Execute();
            }

            public override void MoveWorldObjects()
            {
                if (Owner.Agent == null)
                    return;

                Owner.Controls.SafeCopyToHost();

                MoveAgent(Owner.Agent, Owner.Controls.Host);

                if (Owner.Teacher != null)
                {
                    MoveAgent(Owner.Teacher, Owner.Teacher.CurrentAction());
                }
                
            }

            public void MoveAgent(MovableGameObject a, float[] controls)
            {
                a.vX = a.vY = 0;

                if (!Owner.IsWorldFrozen)
                {
                    if (controls[0] > 0)
                        a.vX += 4;
                    if (controls[1] > 0)
                        a.vX += -4;
                    if (Owner.DegreesOfFreedom == 2)
                    {
                        if (controls[2] > 0)
                            a.vY += 4;
                        if (controls[3] > 0)
                            a.vY += -4;
                    }
                }

                int futureX = a.X + (int)a.vX;
                a.X = futureX;

                int futureY = a.Y + (int)a.vY;
                a.Y = futureY;
            }
        }
    }
}
