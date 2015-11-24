using BEPUphysics;
using BEPUphysics.CollisionRuleManagement;
using BEPUphysics.Constraints.SolverGroups;
using BEPUphysics.Constraints.TwoEntity.Joints;
using BEPUphysics.Constraints.TwoEntity.Motors;
using BEPUphysics.Entities;
using BEPUphysics.Entities.Prefabs;
using BEPUutilities;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Inverted pendulum in 3D World based on BEPUphysics engine.</summary>
    /// <description>Inverted pendulum with one or two controlled joints with one or three degrees of freedom. <br />
    /// Parameters:
    ///              <ul>
    ///                 <li>MOTOR_MODE: Behaviour of joint motor control, VelocityMotor sets target velocity of rotation, ServoMotor sets target rotation</li>
    ///                 <li>ELBOW_FIXED: If true, the second joint (elbow) becomes fixed in its position and uncontrollable</li>
    ///                 <li>POLE_DOF: Degrees of freedom of the uncontrolled joint on the bottom of pole</li>
    ///                 <li>GRAVITY: Gravity strength</li>
    ///              </ul>
    /// I/O:
    ///              <ul>
    ///                 <li>Controls: Control signals for the joints</li>
    ///                 <li>SpherePush: Optional, adds the 3D vector to sphere's velocity</li>
    ///                 <li>Joints: Rotation of the joints, may not be accurate</li>
    ///                 <li>PoleRotation: 1 or 3 dimensional rotation of pole depending on POLE_DOF setting</li>
    ///                 <li>ControlsCopy: Copy of previous control signals</li>
    ///                 <li>SpherePosition: 3D position of the sphere</li>
    ///              </ul>
    /// </description>
    public class My3DPendulumWorld : My3DWorld
    {
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = MotorMode.VelocityMotor)]
        public MotorMode MOTOR_MODE { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = true)]
        public bool ELBOW_FIXED { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 1)]
        public int POLE_DOF { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 9.81f)]
        public float GRAVITY { get; set; }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> SpherePush
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Limits
        {
            get { return GetInput(2); }
        }

        [MyInputBlock(3)]
        public MyMemoryBlock<float> Reset
        {
            get { return GetInput(3); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> PoleRotation
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> ControlsCopy
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> SpherePosition
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        public override void Validate(MyValidator validator)
        {
            //base.Validate(validator);
            validator.AssertError(Controls != null, this, "No input available");

            if (Limits != null)
            {
                validator.AssertError(Limits.Count == 3, this, "Limit vector must be of a size 3 or nothing!");
            }

            if (Reset != null)
            {
                validator.AssertError(Reset.Count == 1, this, "Reset vector must be of a size 1 or nothing!");
            }

            if (Controls != null)
            {
                if (ELBOW_FIXED)
                    validator.AssertError(Controls.Count >= 1, this, "Not enough controls (1 required)");
                else
                    validator.AssertError(Controls.Count >= 2, this, "Not enough controls (2 required)");
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (POLE_DOF == 1)
            {
                Joints.Count = 3;
                PoleRotation.Count = 1;
                ControlsCopy.Count = 1;
            }
            else
            {
                Joints.Count = 5;
                PoleRotation.Count = 3;
                ControlsCopy.Count = 2;
            }

            SpherePosition.Count = 3;
        }

        public My3DWorldTask WorldTask { get; set; }

        /// <summary>Simulates the 3D world</summary>
        [Description("Simulate 3D world")]
        public class My3DWorldTask : MyTask<My3DPendulumWorld>
        {
            private RevoluteJoint shoulder;
            private RevoluteJoint elbow;
            private RevoluteJoint wrist;
            private BallSocketJoint socket;
            private Cylinder pole;
            private Sphere sphere;

            public override void Init(int nGPU)
            {
                CreateWorld();
            }

            private void CreateWorld()
            {
                //Robot arm scene
                Owner.Space = new Space();
                Owner.Space.ForceUpdater.Gravity = new Vector3(0, -Owner.GRAVITY, 0);
                Entity ground = new Box(new Vector3(0, -0.5f, 0), 30, 1, 30);
                Owner.Space.Add(ground);

                var armBase = new Box(new Vector3(0, 0.25f, 0), 1, 0.5f, 1);
                Owner.Space.Add(armBase);

                var lowerArm = new Box(armBase.Position + new Vector3(0, armBase.Height / 2 + 0.75f, 0), 0.5f, 1.5f, 0.5f, 1f);
                Owner.Space.Add(lowerArm);

                var upperArm = new Box(lowerArm.Position + new Vector3(0, lowerArm.Height / 2 + 1, 0), 0.25f, 2, 0.25f, 0.1f);
                Owner.Space.Add(upperArm);

                pole = new Cylinder(upperArm.Position + new Vector3(0, upperArm.Height / 2 + 2, 0), 4, 0.0625f, 0.01f);
                Owner.Space.Add(pole);

                sphere = new Sphere(pole.Position + new Vector3(0, pole.Height / 2, 0), 0.25f, 0.001f);
                Owner.Space.Add(sphere);

                //Lower arm to base joint
                shoulder = new RevoluteJoint(armBase, lowerArm, armBase.Position + new Vector3(0, armBase.Height / 2, 0), Vector3.Forward);
                shoulder.Motor.IsActive = true;
                shoulder.Motor.Settings.Mode = Owner.MOTOR_MODE;
                shoulder.Motor.Settings.MaximumForce = 25;
                shoulder.Limit.IsActive = true;

                float[] limits = GetLimits();

                shoulder.Limit.MinimumAngle = -MathHelper.Pi * limits[2];
                shoulder.Limit.MaximumAngle = MathHelper.Pi * limits[2];
                shoulder.Limit.Bounciness = 0.0f;
                Owner.Space.Add(shoulder);

                //Upper arm to lower arm joint
                elbow = new RevoluteJoint(lowerArm, upperArm, lowerArm.Position + new Vector3(0, lowerArm.Height / 2, 0), Vector3.Left);
                elbow.Motor.IsActive = true;
                if (Owner.ELBOW_FIXED)
                    elbow.Motor.Settings.Mode = MotorMode.Servomechanism;
                else
                    elbow.Motor.Settings.Mode = Owner.MOTOR_MODE;
                elbow.Motor.Settings.MaximumForce = 2500;
                elbow.Limit.IsActive = true;

                elbow.Limit.MinimumAngle = -MathHelper.Pi * limits[1];
                elbow.Limit.MaximumAngle = MathHelper.Pi * limits[1];
                elbow.Limit.Bounciness = 0.0f;
                Owner.Space.Add(elbow);

                if (Owner.POLE_DOF == 1)
                {
                    //Upper arm to pole joint
                    wrist = new RevoluteJoint(upperArm, pole, upperArm.Position + new Vector3(0, upperArm.Height / 2, 0), Vector3.Forward);
                    wrist.Motor.IsActive = false;
                    wrist.Motor.Settings.Mode = Owner.MOTOR_MODE;
                    wrist.Motor.Settings.MaximumForce = 2500;
                    wrist.Limit.IsActive = true;
                    wrist.Limit.MinimumAngle = -MathHelper.Pi * limits[0];
                    wrist.Limit.MaximumAngle = -MathHelper.Pi * limits[0];
                    wrist.Limit.Bounciness = 0.0f;
                    Owner.Space.Add(wrist);
                }
                else
                {
                    socket = new BallSocketJoint(upperArm, pole, upperArm.Position + new Vector3(0, upperArm.Height / 2, 0));
                    Owner.Space.Add(socket);
                }

                //Fixed pole to sphere joint
                var norotation = new RevoluteJoint(pole, sphere, pole.Position + new Vector3(0, pole.Height / 2 - 0.5f, 0), Vector3.Forward);
                norotation.Limit.IsActive = true;
                norotation.Limit.MinimumAngle = 0.0f;
                norotation.Limit.MaximumAngle = 0.0f;

                Owner.Space.Add(norotation);

                CollisionRules.AddRule(armBase, lowerArm, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(lowerArm, upperArm, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(upperArm, pole, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(pole, sphere, CollisionRule.NoBroadPhase);
            }

            public override void Execute()
            {
                if (Owner.Reset != null)
                {
                    Owner.Reset.SafeCopyToHost();
                }

                if (Owner.Reset != null && Owner.Reset.Host[0] > 0.0f)
                {
                    CreateWorld();
                }
                else
                {

                    float dt = Owner.Space.TimeStepSettings.TimeStepDuration;
                    float deadZone = 0.0f;

                    //Pass controls 
                    Owner.Controls.SafeCopyToHost();
                    float[] limits = GetLimits();

                    shoulder.Limit.MinimumAngle = -MathHelper.Pi * limits[2];
                    shoulder.Limit.MaximumAngle = MathHelper.Pi * limits[2];

                    elbow.Limit.MinimumAngle = -MathHelper.Pi * limits[1];
                    elbow.Limit.MaximumAngle = MathHelper.Pi * limits[1];

                    if (wrist != null)
                    {
                        wrist.Limit.MinimumAngle = -MathHelper.Pi * limits[0];
                        wrist.Limit.MaximumAngle = MathHelper.Pi * limits[0];
                    }

                    //shoulder
                    if (Owner.MOTOR_MODE == MotorMode.Servomechanism)
                    {
                        if (Owner.Controls.Host[0] < -deadZone)
                        {
                            shoulder.Motor.Settings.Servo.Goal = MathHelper.Min(shoulder.Motor.Settings.Servo.Goal - Owner.Controls.Host[0] * 0.5f * dt, shoulder.Limit.MaximumAngle);
                        }
                        else if (Owner.Controls.Host[0] > deadZone)
                        {
                            shoulder.Motor.Settings.Servo.Goal = MathHelper.Max(shoulder.Motor.Settings.Servo.Goal - Owner.Controls.Host[0] * 0.5f * dt, shoulder.Limit.MinimumAngle);
                        }
                    }
                    else if (Owner.MOTOR_MODE == MotorMode.VelocityMotor)
                    {
                        shoulder.Motor.Settings.VelocityMotor.GoalVelocity = Owner.Controls.Host[0];
                    }
                    //elbow
                    if (Owner.MOTOR_MODE == MotorMode.Servomechanism)
                    {
                        if (Owner.Controls.Host[1] < -deadZone)
                        {
                            elbow.Motor.Settings.Servo.Goal = MathHelper.Min(elbow.Motor.Settings.Servo.Goal - Owner.Controls.Host[1] * 0.5f * dt, elbow.Limit.MaximumAngle);
                        }
                        else if (Owner.Controls.Host[1] > deadZone)
                        {
                            elbow.Motor.Settings.Servo.Goal = MathHelper.Max(elbow.Motor.Settings.Servo.Goal - Owner.Controls.Host[1] * 0.5f * dt, elbow.Limit.MinimumAngle);
                        }
                    }
                    else if (Owner.MOTOR_MODE == MotorMode.VelocityMotor)
                    {
                        if (Owner.ELBOW_FIXED)
                            elbow.Motor.Settings.Servo.Goal = 0;
                        else
                            elbow.Motor.Settings.VelocityMotor.GoalVelocity = Owner.Controls.Host[1];
                    }

                    //Pushing the sphere with outside force
                    if (Owner.SpherePush != null && Owner.SpherePush.Count >= 3)
                    {
                        Owner.SpherePush.SafeCopyToHost();

                        sphere.LinearVelocity = sphere.LinearVelocity + new Vector3(Owner.SpherePush.Host[0], Owner.SpherePush.Host[1], Owner.SpherePush.Host[2]);
                    }

                    //Update world simulation
                    Owner.Space.Update();
                }


                /*
                //Get feedback
                Owner.Joints.Host[0] = Vector3.Dot(
                    shoulder.Motor.ConnectionA.OrientationMatrix.Up,
                    shoulder.Motor.ConnectionB.OrientationMatrix.Up);
                Owner.Joints.Host[1] = Vector3.Dot(
                    elbow.Motor.ConnectionA.OrientationMatrix.Up,
                    elbow.Motor.ConnectionB.OrientationMatrix.Up);
                //TODO: gravity vector of pole
                Owner.Joints.Host[2] = Vector3.Dot(
                    wrist.Motor.ConnectionA.OrientationMatrix.Up,
                    wrist.Motor.ConnectionB.OrientationMatrix.Up);
                */

                Owner.Joints.Host[0] = shoulder.Motor.ConnectionB.OrientationMatrix.Up.X;
                Owner.Joints.Host[1] = elbow.Motor.ConnectionB.OrientationMatrix.Up.X;

                if (Owner.POLE_DOF == 1)
                {
                    Owner.Joints.Host[2] = wrist.Motor.ConnectionB.OrientationMatrix.Up.X;

                    Owner.PoleRotation.Host[0] = pole.Orientation.Z;
                }
                else
                {
                    Owner.Joints.Host[2] = socket.ConnectionB.OrientationMatrix.Up.X;
                    Owner.Joints.Host[3] = socket.ConnectionB.OrientationMatrix.Up.Y;
                    Owner.Joints.Host[4] = socket.ConnectionB.OrientationMatrix.Up.Z;

                    Owner.PoleRotation.Host[0] = pole.Orientation.Z;
                    Owner.PoleRotation.Host[1] = pole.Orientation.Y;
                    Owner.PoleRotation.Host[2] = pole.Orientation.X;
                }

                Owner.SpherePosition.Host[0] = sphere.Position.X;
                Owner.SpherePosition.Host[1] = sphere.Position.Y;
                Owner.SpherePosition.Host[2] = sphere.Position.Z;

                Owner.Controls.CopyToMemoryBlock(Owner.ControlsCopy, 0, 0, Math.Min(Owner.Controls.Count, Owner.ControlsCopy.Count));
                Owner.Joints.SafeCopyToDevice();
                Owner.PoleRotation.SafeCopyToDevice();
                Owner.SpherePosition.SafeCopyToDevice();
            }

            private float[] GetLimits()
            {
                float[] limits;
                if (Owner.Limits != null)
                {
                    Owner.Limits.SafeCopyToHost();
                    limits = Owner.Limits.Host;
                }
                else
                {
                    limits = new float[] { 0.33f, 0.33f, 0.33f };
                }
                return limits;
            }
        }
    }
}
