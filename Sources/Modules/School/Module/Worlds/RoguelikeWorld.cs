using GoodAI.Core.Nodes;
using GoodAI.Modules.School.Common;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.Worlds
{
	/// <author>GoodAI</author>
    /// <meta>Mp,Mv,Os,Ph,Mm,Ms,Sa</meta>
    /// <status>Working</status>
    /// <summary> Implementation of a configurable top-view 2D world </summary>
    /// <description>
    /// Implementation of a configurable top-view 2D world based on ManInWorld
    /// </description>
    [DisplayName("Roguelike")]
    public partial class RoguelikeWorld : ManInWorld, IMyCustomTaskFactory
    {
        protected override string TEXTURE_DIR { get { return @"res\RoguelikeWorld"; } }

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

        public PointF GetInitPosition()
        {
            SizeF agentPos = Agent == null
                ? RogueAgent.GetDefaultSize()
                : Agent.Size;

            return new PointF((Scene.Width - agentPos.Width) / 2, (Scene.Height - agentPos.Height) / 2);
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

            public override void Execute()
            {
                base.Execute();
            }

            public override void MoveWorldObjects()
            {
                if (Owner.IsWorldFrozen)
                    return;

                if (Owner.Agent == null)
                    return;

                Owner.Controls.SafeCopyToHost();

                MoveAgent(Owner.Agent, Owner.Controls.Host);

                if (Owner.Teacher != null)
                {
                    MoveAgent(Owner.Teacher, Owner.Teacher.CurrentAction());
                }

                base.MoveWorldObjects();
            }

            public void MoveAgent(MovableGameObject a, float[] controls)
            {
                a.Velocity.X = a.Velocity.Y = 0;

                if (!Owner.IsWorldFrozen)
                {
                    if (controls[0] > 0)
                        a.Velocity.X += 4;
                    if (controls[1] > 0)
                        a.Velocity.X += -4;
                    if (Owner.DegreesOfFreedom == 2)
                    {
                        if (controls[2] > 0)
                            a.Velocity.Y += 4;
                        if (controls[3] > 0)
                            a.Velocity.Y += -4;
                    }
                }

                //int futureX = a.X + (int)a.vX;
                //a.X = futureX;

                //int futureY = a.Y + (int)a.vY;
                //a.Y = futureY;
            }
        }
    }
}
