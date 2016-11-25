using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using System;
using System.ComponentModel;
using System.Drawing;
using YAXLib;

namespace GoodAI.Modules.School.Worlds
{
    /// <author>GoodAI</author>
    /// <meta>os</meta>
    /// <status>WIP</status>
    /// <summary> Custom implementation of the plumber world (Mario) </summary>
    /// <description>
    /// 2D Platformer world which includes a traditional 2D physics engine and collision detection/handling
    /// </description>
    [DisplayName("Plumber")]
    public partial class PlumberWorld : ManInWorld, IMyCustomTaskFactory
    {
        // temporary implementation for actions
        public float moveLeftAction;
        public float moveRightAction;
        public float moveUpAction;
        public float moveDownAction;
        protected GameObject GameObjectInControl = null;    //This denotes the reference to the GameObject currently controlled by the User

        protected override string TEXTURE_DIR { get { return @"res\PlumberWorld"; } }


        public PlumberWorld()  // Constructor method
        {
            Scene = new SizeF(1000, 300);
            Viewport = new SizeF(300, 200);
        }


        public override MovableGameObject CreateAgent(string iconPath, PointF position)
        {
            base.CreateAgent(iconPath, position);
            GameObjectInControl = Agent;
            return Agent;
        }

        public virtual void CreateTasks()
        {
            //InitSchool = new InitialiseWorldTask();
            GetInputTask = new GetPlumberInputTask();
            UpdateWorldTask = new UpdatePlumberTask();
            RenderGLWorldTask = new RenderGLTask();
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Controls != null, this, "Controls must not be null");
            if (Controls != null)
                validator.AssertError(Controls.Count >= 3, this, "Size of Control input must be 3 or more");
        }


        public class GetPlumberInputTask : InputTask
        {
            public PlumberWorld PlumberOwner { get { return (PlumberWorld)Owner; } }

            public override void Init(int nGPU)
            {
                base.Init(nGPU);
            }

            public override void Execute()
            {
                base.Execute();

                //Temporary way of defining input for testing the 2D physics
                if (!Owner.IsWorldFrozen)
                {
                    PlumberOwner.Controls.SafeCopyToHost();
                    PlumberOwner.moveUpAction = PlumberOwner.Controls.Host[0];
                    PlumberOwner.moveRightAction = PlumberOwner.Controls.Host[1];
                    PlumberOwner.moveLeftAction = PlumberOwner.Controls.Host[2];
                }
                else
                {
                    PlumberOwner.moveUpAction = 0;
                    PlumberOwner.moveRightAction = 0;
                    PlumberOwner.moveLeftAction = 0;
                }

                //each action is either 1 or 0 (Button pressed or not)
                PlumberOwner.moveUpAction = (float)Math.Round(PlumberOwner.moveUpAction);
                PlumberOwner.moveRightAction = (float)Math.Round(PlumberOwner.moveRightAction);
                PlumberOwner.moveLeftAction = (float)Math.Round(PlumberOwner.moveLeftAction);
            }
        }

        /// <summary>
        /// Initialises The objects that will populate the world using "AddGameObject" which is derived from ManInWorld world, this will define the current level
        /// </summary>
        //[MyTaskInfo(OneShot = true, Order = 90)]
        //public class InitialiseWorldTask : InitSchoolTask
        //{
        //    public PlumberWorld PlumberOwner { get { return (PlumberWorld)Owner; } }

        //    public override void Init(int nGPU)
        //    {
        //        base.Init(nGPU);
        //    }

        //    public override void Execute()
        //    {
        //        base.Execute();
        //    }
        //}

        /// <summary>
        /// Apply configurable gravity physics to the specified GameObjects
        /// </summary>
        public class UpdatePlumberTask : UpdateTask
        {
            public PlumberWorld PlumberOwner { get { return (PlumberWorld)Owner; } }

            [MyBrowsable, Category("Gravity Params"),
            Description("Gravitational acceleration")]
            [YAXSerializableField(DefaultValue = 0.218f)]
            public float GravityAcc { get; set; }

            [MyBrowsable, Category("Gravity Params"),
            Description("How powerful jumps are. Defines vertical boost to the velocity of the player when jumps are detected, negative value required")]
            [YAXSerializableField(DefaultValue = -7f)]
            public float JumpBoost { get; set; }

            [MyBrowsable, Category("Physics Params"),
            Description("How fast agent can move on X axis")]
            [YAXSerializableField(DefaultValue = 8.0f)]
            public float MaximumAgentSpeed { get; set; }

            [MyBrowsable, Category("Physics Params"),
            Description("How fast objects can fall")]
            [YAXSerializableField(DefaultValue = 8.0f)]
            public float MaximumFallSpeed { get; set; }

            [MyBrowsable, Category("Physics Params"),
            Description("How fast agent accelerates on X axis while directional button is pressed")]
            [YAXSerializableField(DefaultValue = 0.1f)]
            public float Acceleration { get; set; }

            [MyBrowsable, Category("Physics Params"),
            Description("How deceleration affects movable objects, reasonable values range from 0(don't decelerate) to 0.1 (decelerate heavily), 1 means sudden halt")]
            [YAXSerializableField(DefaultValue = 0.03f)]
            public float Deceleration { get; set; }

            public override void Init(int nGPU)
            {
                base.Init(nGPU);
            }

            public void ApplyGravity()
            {
                if (Owner.IsWorldFrozen)
                    return;

                // Compute Gravity step for all objects affected by physics
                for (int i = 0; i < PlumberOwner.GameObjects.Count; i++)
                {
                    if (PlumberOwner.GameObjects[i] is MovableGameObject)
                    {
                        GameObject obj = Owner.GameObjects[i];
                        MovableGameObject mobj = obj as MovableGameObject;

                        if (mobj.IsAffectedByGravity)
                        {
                            mobj.Velocity.Y += GravityAcc * Owner.Time;        // Apply gravity to vertical velocity
                            if (mobj.Velocity.Y > MaximumFallSpeed)
                            {
                                mobj.Velocity.Y = MaximumFallSpeed;
                            }
                        }
                    }
                }
            }

            public override void MoveWorldObjects()
            {
                ApplyGravity();

                // Updates agent's speed vector based on input
                // Updates game objects by gravity

                MovableGameObject magent = (PlumberOwner.GameObjectInControl as MovableGameObject);   //Create reference to the GameObject currently under User control, we need to know its OnGround value
                if (magent == null)
                    return;

                if (PlumberOwner.moveUpAction == 1.0f && magent.OnGround == true)
                {
                    magent.OnGround = false;                                // If the jump command request was detected, give boost to the Y velocity of the Object in control (jump)
                    magent.Velocity.Y = JumpBoost;
                }
                if (PlumberOwner.moveRightAction == 1.0f)
                {
                    magent.Velocity.X += Acceleration;                                  // Apply acceleration while "right button" is pressed
                    if (magent.Velocity.X > MaximumAgentSpeed)                          // Limit Max velocity
                    {
                        magent.Velocity.X = MaximumAgentSpeed;
                    }
                }
                if (PlumberOwner.moveLeftAction == 1.0f)
                {
                    magent.Velocity.X -= Acceleration;                                  // Apply acceleration while "left button" is pressed
                    if (magent.Velocity.X < -MaximumAgentSpeed)                         // Limit Max velocity
                    {
                        magent.Velocity.X = -MaximumAgentSpeed;
                    }
                }

                if (PlumberOwner.moveLeftAction == 0.0f && PlumberOwner.moveRightAction == 0.0f)
                {
                    magent.Velocity.X *= (1 - Deceleration);                              // Decelerate when no corresponding directional button is not pressed
                }

                base.MoveWorldObjects();

                /*
                MyLog.DEBUG.WriteLine("===============================");
                MyLog.DEBUG.WriteLine("previousX: " + magent.previousX);
                MyLog.DEBUG.WriteLine("previousY: " + magent.previousY);
                MyLog.DEBUG.WriteLine("X: " + PlumberOwner.GameObjectInControl.X);
                MyLog.DEBUG.WriteLine("Y: " + PlumberOwner.GameObjectInControl.Y);
                MyLog.DEBUG.WriteLine("previousVelocityX: " + magent.previousvX);
                MyLog.DEBUG.WriteLine("previousVelocityY: " + magent.previousvY);
                MyLog.DEBUG.WriteLine("VelocityX: " + magent.vX);
                MyLog.DEBUG.WriteLine("VelocityY: " + magent.vY);
                MyLog.DEBUG.WriteLine("OnGround: " + magent.OnGround);
                */
            }


            public override void Execute()
            {
                base.Execute();
            }
        }
    }
}
