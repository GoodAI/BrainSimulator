using BEPUphysics;
using BEPUphysics.BroadPhaseEntries.MobileCollidables;
using BEPUphysics.CollisionRuleManagement;
using BEPUphysics.CollisionShapes;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysics.Constraints.SolverGroups;
using BEPUphysics.Constraints.TwoEntity.Motors;
using BEPUphysics.Entities;
using BEPUphysics.Entities.Prefabs;
using BEPUphysics.Materials;
using BEPUutilities;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>Manipulator in 3D World based on BEPUphysics engine</summary>
    /// <description>Manipulator from BEPUphysics demo "Robotic Arm Thingamajig" with 5 rotational joints. <br />
    /// I/O:
    ///              <ul>
    ///                 <li>Controls: Control signals for the joints</li>
    ///                 <li>Joints: Rotation of the joints, may not be accurate</li>
    ///              </ul>
    /// </description>
    public class My3DManipulatorWorld : My3DWorld
    {

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (Controls != null)
            {
                validator.AssertError(Controls.Count >= 5, this, "Not enough controls, 5 needed!");
            }
        }

        public override void UpdateMemoryBlocks()
        {
            Joints.Count = 6;
        }

        public My3DWorldTask WorldTask { get; set; }

        /// <summary>Simulates the 3D world</summary>
        [Description("Simulate 3D world")]
        public class My3DWorldTask : MyTask<My3DWorld>
        {
            private RevoluteJoint groundToBaseJoint;
            private RevoluteJoint shoulder;
            private SwivelHingeJoint elbow;
            private RevoluteJoint clawHingeA;
            private RevoluteJoint clawHingeB;

            public override void Init(int nGPU)
            {
                //Testing scene

                /*                
                Owner.Space = new Space();
                Owner.Space.ForceUpdater.Gravity = new Vector3(0, -9.81f, 0);

                Box ground = new Box(new Vector3(0, -0.55f, 0), 30, 1, 30);
                Owner.Space.Add(ground);                

                Owner.Space.Add(new Box(new Vector3(0.3f, 4, 0), 1, 1, 1, 1));
                Owner.Space.Add(new Box(new Vector3(0, 8, 0.3f), 1, 1, 1, 1));
                Owner.Space.Add(new Box(new Vector3(-0.3f, 12, 0), 1, 1, 1, 1));

                Owner.Space.Add(new Cylinder(new Vector3(0.7f, 15, 0), 2, 1, 1));

                Owner.Space.Add(new Cone(new Vector3(-0.5f, 18, 0.3f), 2, 1, 1));

                Owner.Space.Add(new Sphere(new Vector3(2, 10, 0), 1, 1));
                 * */

                //Robot arm scene

                Owner.Space = new Space();
                Owner.Space.ForceUpdater.Gravity = new Vector3(0, -9.81f, 0);
                Entity ground = new Box(Vector3.Zero, 30, 1, 30);
                Owner.Space.Add(ground);

                var armBase = new Cylinder(new Vector3(0, 2, 0), 2.5f, 1, 40);
                Owner.Space.Add(armBase);

                //The arm base can rotate around the Y axis.
                //Rotation is controlled by user input.
                groundToBaseJoint = new RevoluteJoint(ground, armBase, Vector3.Zero, Vector3.Up);
                groundToBaseJoint.Motor.IsActive = true;
                groundToBaseJoint.Motor.Settings.Mode = MotorMode.Servomechanism;
                groundToBaseJoint.Motor.Settings.MaximumForce = 3500;
                Owner.Space.Add(groundToBaseJoint);

                Entity lowerArm = new Box(armBase.Position + new Vector3(0, 2, 0), 1, 3, .5f, 10);
                Owner.Space.Add(lowerArm);

                shoulder = new RevoluteJoint(armBase, lowerArm, armBase.Position, Vector3.Forward);
                shoulder.Motor.IsActive = true;
                shoulder.Motor.Settings.Mode = MotorMode.Servomechanism;
                shoulder.Motor.Settings.MaximumForce = 2500;

                //Don't want it to rotate too far; this keeps the whole contraption off the ground.
                shoulder.Limit.IsActive = true;
                shoulder.Limit.MinimumAngle = -MathHelper.Pi / 3.0f;
                shoulder.Limit.MaximumAngle = MathHelper.Pi / 3.0f;
                Owner.Space.Add(shoulder);

                //Make the next piece of the arm.
                Entity upperArm = new Cylinder(lowerArm.Position + new Vector3(0, 3, 0), 3, .25f, 10);
                Owner.Space.Add(upperArm);

                //Swivel hinges allow motion around two axes.  Imagine a tablet PC's monitor hinge.
                elbow = new SwivelHingeJoint(lowerArm, upperArm, lowerArm.Position + new Vector3(0, 1.5f, 0), Vector3.Forward);
                elbow.TwistMotor.IsActive = true;
                elbow.TwistMotor.Settings.Mode = MotorMode.Servomechanism;
                elbow.TwistMotor.Settings.MaximumForce = 1000;

                elbow.HingeMotor.IsActive = true;
                elbow.HingeMotor.Settings.Mode = MotorMode.Servomechanism;
                elbow.HingeMotor.Settings.MaximumForce = 1250;

                //Keep it from rotating too much.
                elbow.HingeLimit.IsActive = true;
                elbow.HingeLimit.MinimumAngle = -MathHelper.PiOver2;
                elbow.HingeLimit.MaximumAngle = MathHelper.PiOver2;
                Owner.Space.Add(elbow);


                //Add a menacing claw at the end.
                var lowerPosition = upperArm.Position + new Vector3(-.65f, 1.6f, 0);

                CollisionRules clawPart1ARules = new CollisionRules();
                var bodies = new List<CompoundChildData>()
                {
                    new CompoundChildData() { Entry = new CompoundShapeEntry(new BoxShape(1, .25f, .25f), lowerPosition, 3), CollisionRules = clawPart1ARules },
                    new CompoundChildData() { Entry = new CompoundShapeEntry(new ConeShape(1, .125f), lowerPosition + new Vector3(-.375f, .4f, 0), 3), Material = new Material(2,2,0) }
                };

                var claw = new CompoundBody(bodies, 6);
                Owner.Space.Add(claw);

                clawHingeA = new RevoluteJoint(upperArm, claw, upperArm.Position + new Vector3(0, 1.5f, 0), Vector3.Forward);
                clawHingeA.Motor.IsActive = true;
                clawHingeA.Motor.Settings.Mode = MotorMode.Servomechanism;
                clawHingeA.Motor.Settings.Servo.Goal = -MathHelper.PiOver2;
                //Weaken the claw to prevent it from crushing the boxes.
                clawHingeA.Motor.Settings.Servo.SpringSettings.Damping /= 100;
                clawHingeA.Motor.Settings.Servo.SpringSettings.Stiffness /= 100;

                clawHingeA.Limit.IsActive = true;
                clawHingeA.Limit.MinimumAngle = -MathHelper.PiOver2;
                clawHingeA.Limit.MaximumAngle = -MathHelper.Pi / 6;
                Owner.Space.Add(clawHingeA);

                //Add one more claw to complete the contraption.
                lowerPosition = upperArm.Position + new Vector3(.65f, 1.6f, 0);

                CollisionRules clawPart1BRules = new CollisionRules();
                bodies = new List<CompoundChildData>()
                {
                    new CompoundChildData() { Entry = new CompoundShapeEntry(new BoxShape(1, .25f, .25f), lowerPosition, 3), CollisionRules = clawPart1BRules },
                    new CompoundChildData() { Entry = new CompoundShapeEntry(new ConeShape(1, .125f), lowerPosition + new Vector3(.375f, .4f, 0), 3), Material = new Material(2,2,0) }
                };
                claw = new CompoundBody(bodies, 6);
                Owner.Space.Add(claw);

                clawHingeB = new RevoluteJoint(upperArm, claw, upperArm.Position + new Vector3(0, 1.5f, 0), Vector3.Forward);
                clawHingeB.Motor.IsActive = true;
                clawHingeB.Motor.Settings.Mode = MotorMode.Servomechanism;
                clawHingeB.Motor.Settings.Servo.Goal = MathHelper.PiOver2;
                //Weaken the claw to prevent it from crushing the boxes.
                clawHingeB.Motor.Settings.Servo.SpringSettings.Damping /= 100;
                clawHingeB.Motor.Settings.Servo.SpringSettings.Stiffness /= 100;

                clawHingeB.Limit.IsActive = true;
                clawHingeB.Limit.MinimumAngle = MathHelper.Pi / 6;
                clawHingeB.Limit.MaximumAngle = MathHelper.PiOver2;
                Owner.Space.Add(clawHingeB);

                //Keep the pieces of the robot from interacting with each other.
                //The CollisionRules.AddRule method is just a convenience method that adds items to the 'specific' dictionary.
                //Sometimes, it's a little unwieldy to reference the collision rules,
                //so the convenience method just takes the owners and hides the ugly syntax.
                CollisionRules.AddRule(armBase, lowerArm, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(lowerArm, upperArm, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(clawPart1ARules, upperArm, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(clawPart1BRules, upperArm, CollisionRule.NoBroadPhase);
                //Here's an example without a convenience method.  Since they are both direct CollisionRules references, it's pretty clean.
                clawPart1BRules.Specific.Add(clawPart1ARules, CollisionRule.NoBroadPhase);

                //Put some boxes on the ground to try to pick up.
                for (double k = 0; k < Math.PI * 2; k += Math.PI / 6)
                {
                    Box box = new Box(new Vector3((float)Math.Cos(k) * 5.5f, 2, (float)Math.Sin(k) * 5.5f), 1, 1, 1, 10);
                    box.Tag = Color.LightCoral;

                    Owner.Space.Add(box);
                }
            }

            public override void Execute()
            {
                float dt = Owner.Space.TimeStepSettings.TimeStepDuration;
                float deadZone = 0.1f;

                //Pass controls 
                Owner.Controls.SafeCopyToHost();

                if (Owner.Controls.Host[0] < -deadZone)
                {
                    groundToBaseJoint.Motor.Settings.Servo.Goal += Owner.Controls.Host[0] * 0.5f * dt;
                }
                else if (Owner.Controls.Host[0] > deadZone)
                {
                    groundToBaseJoint.Motor.Settings.Servo.Goal += Owner.Controls.Host[0] * 0.5f * dt;
                }

                if (Owner.Controls.Host[1] < -deadZone)
                {
                    shoulder.Motor.Settings.Servo.Goal = MathHelper.Min(shoulder.Motor.Settings.Servo.Goal - Owner.Controls.Host[1] * 0.5f * dt, shoulder.Limit.MaximumAngle);
                }
                else if (Owner.Controls.Host[1] > deadZone)
                {
                    shoulder.Motor.Settings.Servo.Goal = MathHelper.Max(shoulder.Motor.Settings.Servo.Goal - Owner.Controls.Host[1] * 0.5f * dt, shoulder.Limit.MinimumAngle);
                }

                if (Owner.Controls.Host[2] < -deadZone)
                {
                    elbow.HingeMotor.Settings.Servo.Goal = MathHelper.Min(elbow.HingeMotor.Settings.Servo.Goal - Owner.Controls.Host[2] * 0.5f * dt, elbow.HingeLimit.MaximumAngle);
                }
                else if (Owner.Controls.Host[2] > deadZone)
                {
                    elbow.HingeMotor.Settings.Servo.Goal = MathHelper.Max(elbow.HingeMotor.Settings.Servo.Goal - Owner.Controls.Host[2] * 0.5f * dt, elbow.HingeLimit.MinimumAngle);
                }


                if (Owner.Controls.Host[3] < -deadZone)
                {
                    elbow.TwistMotor.Settings.Servo.Goal -= Owner.Controls.Host[3] * dt;
                }
                else if (Owner.Controls.Host[3] > deadZone)
                {
                    elbow.TwistMotor.Settings.Servo.Goal -= Owner.Controls.Host[3] * dt;
                }

                if (Owner.Controls.Host[4] < -deadZone)
                {
                    clawHingeA.Motor.Settings.Servo.Goal = MathHelper.Max(clawHingeA.Motor.Settings.Servo.Goal + Owner.Controls.Host[4] * 1.5f * dt, clawHingeA.Limit.MinimumAngle);
                    clawHingeB.Motor.Settings.Servo.Goal = MathHelper.Min(clawHingeB.Motor.Settings.Servo.Goal - Owner.Controls.Host[4] * 1.5f * dt, clawHingeB.Limit.MaximumAngle);
                }
                else if (Owner.Controls.Host[4] > deadZone)
                {
                    clawHingeA.Motor.Settings.Servo.Goal = MathHelper.Min(clawHingeA.Motor.Settings.Servo.Goal + Owner.Controls.Host[4] * 1.5f * dt, clawHingeA.Limit.MaximumAngle);
                    clawHingeB.Motor.Settings.Servo.Goal = MathHelper.Max(clawHingeB.Motor.Settings.Servo.Goal - Owner.Controls.Host[4] * 1.5f * dt, clawHingeB.Limit.MinimumAngle);
                }

                //Update world simulation
                Owner.Space.Update();

                //Get feedback
                Owner.Joints.Host[0] = Vector3.Dot(
                    groundToBaseJoint.Motor.ConnectionA.OrientationMatrix.Forward,
                    groundToBaseJoint.Motor.ConnectionB.OrientationMatrix.Forward);

                Owner.Joints.Host[1] = Vector3.Dot(
                    shoulder.Motor.ConnectionA.OrientationMatrix.Up,
                    shoulder.Motor.ConnectionB.OrientationMatrix.Up);

                Owner.Joints.Host[2] = Vector3.Dot(
                    elbow.HingeMotor.ConnectionA.OrientationMatrix.Up,
                    elbow.HingeMotor.ConnectionB.OrientationMatrix.Up);

                Owner.Joints.Host[3] = Vector3.Dot(
                    elbow.TwistMotor.ConnectionA.OrientationMatrix.Forward,
                    elbow.TwistMotor.ConnectionB.OrientationMatrix.Forward);

                Owner.Joints.Host[4] = Vector3.Dot(
                    clawHingeA.Motor.ConnectionA.OrientationMatrix.Left,
                    clawHingeA.Motor.ConnectionB.OrientationMatrix.Left);

                Owner.Joints.Host[5] = Vector3.Dot(
                    clawHingeB.Motor.ConnectionA.OrientationMatrix.Left,
                    clawHingeB.Motor.ConnectionB.OrientationMatrix.Left);


                Owner.Joints.SafeCopyToDevice();
            }
        }
    }
}
