using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace GoodAI.Modules.School.Worlds
{
    public partial class RoguelikeWorld : ManInWorld, IMyCustomTaskFactory
    {
        public static Size DEFAULT_GRID_SIZE = new Size(32, 32);

        protected override string TEXTURE_DIR { get { return @"res\RoguelikeWorld"; } }

        //[MyTaskInfo(OneShot = true, Order = 90)]
        //public class InitTask : InitSchoolTask
        //{
        //    public override void Init(int nGPU)
        //    {
        //        base.Init(nGPU);
        //    }

        //    public override void Execute()
        //    {
        //        base.Execute();
        //    }
        //}

        public override void Validate(Core.Utils.MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Controls != null, this, "Controls must not be null");
            if (Controls != null)
                validator.AssertError(Controls.Count >= 4, this, "Size of Controls must be 4 or more");
        }

        public virtual void CreateTasks()
        {
            //InitSchool = new InitTask();
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

        public override Grid GetGrid()
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
