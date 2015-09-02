using BEPUphysics;
using BEPUphysics.CollisionRuleManagement;
using BEPUphysics.Constraints.SolverGroups;
using BEPUphysics.Constraints.TwoEntity.Motors;
using BEPUphysics.Entities;
using BEPUphysics.Entities.Prefabs;
using BEPUutilities;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Motor
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Bipedal robot in 3D world based on BEPUphysics engine.</summary>
    /// <description>Bipedal robot with 9 controlled joints. <br />
    /// Parameters:
    ///              <ul>
    ///                 <li>MOTOR_MODE: Behaviour of joint motor control: VELOCITY_MOTOR sets target velocity of rotation; SERVO sets target rotation; MUSCLE_PAIR requires two controls per joint, for flexor and extensor force</li>
    ///                 <li>MOTOR_MAX_FORCE: Maximu force  motor can produce, doesn't apply for MUSCLE_PAIR motor mode</li>
    ///                 <li>JOINT_STIFFNESS: Stiffnes coefficient of spring in the joint</li>
    ///                 <li>GROUND: Toggle ground for robot to stand on</li>
    ///                 <li>GRAVITY: Gravity strength</li>
    ///                 <li>*_WEIGHT: Weight of the respective body part</li>
    ///                 <li>*_WIDTH or *_HEIGHT: Dimension of the respective body part</li>
    ///                 <li>FORCE_MULTIPLIER: Applies only to MUSCLE_PAIR motor mode, coefficient for flexor/extensor strength</li>
    ///                 <li>VELOCITY_MULTIPLIER: Applies only to MUSCLE_PAIR motor mode, coefficient for flexor/extensor target velocities</li>
    ///              </ul>
    /// I/O:
    ///              <ul>
    ///                 <li>Controls: Control signals for the joints</li>
    ///                 <li>Push: Optional, adds the 3D vector to torso's velocity</li>
    ///                 <li>Joints: Rotation of the joints, may not be accurate</li>
    ///                 <li>TorsoRotation: Rotation of torso in 3D coordinates</li>
    ///                 <li>FeetPressure: Approximate pressure applied to rear part of left foot, front part of left foot, rear part of right foot, front part of right foot in that order</li>
    ///                 <li>ControlsCopy: Copy of previous control signals</li>
    ///                 <li>CenterOfMass: Position of center of mass in 3D coordinates</li>
    ///                 <li>ContactPoint: Position of center of pressure in 3D coordinates</li>
    ///                 <li>JointPosition: 3xN matrix of positions of all joints, each row containing a 3D position of respective joint</li>
    ///                 <li>JointAxis: 3xN matrix of axis of joint rotation, each row containing a 3D axis of respective joint</li>
    ///                 <li>TorsoPosition: Position of torso in 3D coordinates</li>
    ///              </ul>
    /// </description>
    public class My3DBipedalRobotWorld : My3DWorld
    {
        public enum MyMotorMode
        {
            VELOCITY_MOTOR,
            SERVO,
            MUSCLE_PAIR
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = MyMotorMode.VELOCITY_MOTOR)]
        public MyMotorMode MOTOR_MODE { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 100)]
        public float MOTOR_MAX_FORCE { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 100)]
        public float JOINT_STIFFNESS { get; set; }

        public int JOINTS = 9;

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Push
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> TorsoRotation
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> FeetPressure
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> ControlsCopy
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> CenterOfMass
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        [MyOutputBlock(5)]
        public MyMemoryBlock<float> CenterOfPressure
        {
            get { return GetOutput(5); }
            set { SetOutput(5, value); }
        }

        [MyOutputBlock(6)]
        public MyMemoryBlock<float>JointPosition
        {
            get { return GetOutput(6); }
            set { SetOutput(6, value); }
        }

        [MyOutputBlock(7)]
        public MyMemoryBlock<float> JointAxis
        {
            get { return GetOutput(7); }
            set { SetOutput(7, value); }
        }

        [MyOutputBlock(8)]
        public MyMemoryBlock<float> TorsoPosition
        {
            get { return GetOutput(8); }
            set { SetOutput(8, value); }
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(Controls != null && Controls.Count >= (MOTOR_MODE == MyMotorMode.MUSCLE_PAIR ? 2 * JOINTS : JOINTS), this, "Not enough controls");
        }

        public override void UpdateMemoryBlocks()
        {
            Joints.Count = JOINTS;
            TorsoRotation.Count = 3;
            FeetPressure.Count = 4;
            ControlsCopy.Count = (MOTOR_MODE == MyMotorMode.MUSCLE_PAIR ? 2 * JOINTS : JOINTS);
            CenterOfMass.Count = 3;
            CenterOfPressure.Count = 3;
            JointPosition.Count = JOINTS * 3;
            JointPosition.ColumnHint = 3;
            JointAxis.Count = JOINTS * 3;
            JointAxis.ColumnHint = 3;
            TorsoPosition.Count = 3;
        }

        public My3DWorldTask WorldTask { get; set; }

        /// <summary>Simulates the 3D world</summary>
        [Description("Simulate 3D world")]
        public class My3DWorldTask : MyTask<My3DBipedalRobotWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = true)]
            public bool GROUND { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 9.81f)]
            public float GRAVITY { get; set; }

            #region Bodypart weight constants
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 5.0f)]
            public float TORSO_WEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 4.0f)]
            public float PELVIS_WEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 3.0f)]
            public float UPPER_THIGH_WEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 2.0f)]
            public float THIGH_WEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float CALF_WEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float FOOT_WEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float FRONT_FOOT_WEIGHT { get; set; }
            #endregion

            #region Bodypart size constants
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.6f)]
            public float UPPER_THIGH_HEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float THIGH_WIDTH { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 2.0f)]
            public float THIGH_HEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.4f)]
            public float CALF_WIDTH { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 2.0f)]
            public float CALF_HEIGHT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 2.0f)]
            public float PELVIS_HEIGHT { get; set; }
            #endregion

            #region Muscle motor constants
            [MyBrowsable, Category("Muscles")]
            [YAXSerializableField(DefaultValue = 5.0f)]
            public float FORCE_MULTIPLIER { get; set; }

            [MyBrowsable, Category("Muscles")]
            [YAXSerializableField(DefaultValue = 3.0f)]
            public float VELOCITY_MULTIPLIER { get; set; }
            #endregion

            private Cylinder pelvis;
            private Box leftUpperThigh;
            private Box rightUpperThigh;
            private RevoluteJoint leftLateralHip;
            private RevoluteJoint rightLateralHip;
            private Box leftThigh;
            private Box rightThigh;
            private RevoluteJoint leftHip;
            private RevoluteJoint rightHip;
            private Box leftCalf;
            private Box rightCalf;
            private RevoluteJoint leftKnee;
            private RevoluteJoint rightKnee;
            private Box leftFoot;
            private Box rightFoot;
            private RevoluteJoint leftAnkle;
            private RevoluteJoint rightAnkle;
            private Box torso;
            private RevoluteJoint lowerSpineJoint;
            private Box leftFrontFoot;
            private Box rightFrontFoot;

            private List<Entity> bodyparts;
            private List<RevoluteJoint> joints;

            public override void Init(int nGPU)
            {
                Owner.Space = new Space();
                Owner.Space.ForceUpdater.Gravity = new Vector3(0, -GRAVITY, 0);

                if (GROUND)
                {
                    Entity ground = new Box(new Vector3(0, -0.5f, 0), 30, 1, 30);
                    Owner.Space.Add(ground);
                }

                leftFoot = new Box(new Vector3(0.05f, 0.15f / 2, -PELVIS_HEIGHT / 2 + THIGH_WIDTH / 2), 0.6f, 0.15f, 0.3f, FOOT_WEIGHT);
                rightFoot = new Box(new Vector3(0.05f, 0.15f / 2, PELVIS_HEIGHT / 2 - THIGH_WIDTH / 2), 0.6f, 0.15f, 0.3f, FOOT_WEIGHT);

                leftFrontFoot = new Box(leftFoot.Position + new Vector3(0.15f + leftFoot.HalfWidth, 0, 0), 0.3f, 0.15f, 0.3f, FRONT_FOOT_WEIGHT);
                rightFrontFoot = new Box(rightFoot.Position + new Vector3(0.15f + rightFoot.HalfWidth, 0, 0), 0.3f, 0.15f, 0.3f, FRONT_FOOT_WEIGHT);

                leftCalf = new Box(new Vector3(0, leftFoot.Position.Y + CALF_HEIGHT / 2 + 0.1f, -PELVIS_HEIGHT / 2 + THIGH_WIDTH / 2), CALF_WIDTH, CALF_HEIGHT, CALF_WIDTH, CALF_WEIGHT);
                rightCalf = new Box(new Vector3(0, rightFoot.Position.Y + CALF_HEIGHT / 2 + 0.1f, PELVIS_HEIGHT / 2 - THIGH_WIDTH / 2), CALF_WIDTH, CALF_HEIGHT, CALF_WIDTH, CALF_WEIGHT);
                leftThigh = new Box(new Vector3(0, leftCalf.Position.Y + leftCalf.HalfHeight + THIGH_HEIGHT / 2, -PELVIS_HEIGHT / 2 + THIGH_WIDTH / 2), THIGH_WIDTH, THIGH_HEIGHT, THIGH_WIDTH, THIGH_WEIGHT);
                rightThigh = new Box(new Vector3(0, rightCalf.Position.Y + rightCalf.HalfHeight + THIGH_HEIGHT / 2, PELVIS_HEIGHT / 2 - THIGH_WIDTH / 2), THIGH_WIDTH, THIGH_HEIGHT, THIGH_WIDTH, THIGH_WEIGHT);
                leftUpperThigh = new Box(new Vector3(0, leftThigh.Position.Y + leftThigh.HalfHeight + UPPER_THIGH_HEIGHT / 2, -PELVIS_HEIGHT / 2 + THIGH_WIDTH / 2), THIGH_WIDTH, UPPER_THIGH_HEIGHT, THIGH_WIDTH, UPPER_THIGH_WEIGHT);
                rightUpperThigh = new Box(new Vector3(0, leftThigh.Position.Y + leftThigh.HalfHeight + UPPER_THIGH_HEIGHT / 2, PELVIS_HEIGHT / 2 - THIGH_WIDTH / 2), THIGH_WIDTH, UPPER_THIGH_HEIGHT, THIGH_WIDTH, UPPER_THIGH_WEIGHT);

                pelvis = new Cylinder(new Vector3(0, leftUpperThigh.Position.Y + leftUpperThigh.HalfHeight, 0), 2, 0.5f, PELVIS_WEIGHT);
                pelvis.Orientation = new Quaternion(0, (float)Math.PI / 2, (float)Math.PI / 2, 0);
                torso = new Box(new Vector3(0, pelvis.Position.Y + pelvis.Radius / 2 + 1, 0), 0.5f, 2, 0.5f, TORSO_WEIGHT);

                leftLateralHip = new RevoluteJoint(pelvis, leftUpperThigh, new Vector3(0, pelvis.Position.Y - pelvis.Radius / 2, -pelvis.Height / 2 + UPPER_THIGH_HEIGHT / 2), Vector3.Left);
                SetJointMotor(leftLateralHip, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(leftLateralHip, -MathHelper.Pi / 6, MathHelper.Pi / 6);

                rightLateralHip = new RevoluteJoint(pelvis, rightUpperThigh, new Vector3(0, pelvis.Position.Y - pelvis.Radius / 2, pelvis.Height / 2 - UPPER_THIGH_HEIGHT / 2), Vector3.Left);
                SetJointMotor(rightLateralHip, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(rightLateralHip, -MathHelper.Pi / 6, MathHelper.Pi / 6);

                leftHip = new RevoluteJoint(leftUpperThigh, leftThigh, new Vector3(0, leftUpperThigh.Position.Y - leftUpperThigh.HalfHeight, -pelvis.Height / 2 + THIGH_WIDTH / 2), Vector3.Forward);
                SetJointMotor(leftHip, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(leftHip, -0.7f * MathHelper.Pi, 0.1f * MathHelper.Pi);

                rightHip = new RevoluteJoint(rightUpperThigh, rightThigh, new Vector3(0, leftUpperThigh.Position.Y - leftUpperThigh.HalfHeight, pelvis.Height / 2 - THIGH_WIDTH / 2), Vector3.Forward);
                SetJointMotor(rightHip, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(rightHip, -0.7f * MathHelper.Pi, 0.1f * MathHelper.Pi);

                leftKnee = new RevoluteJoint(leftThigh, leftCalf, new Vector3(0, leftThigh.Position.Y - leftThigh.HalfHeight + 0.1f, -pelvis.Height / 2 + THIGH_WIDTH / 2), Vector3.Forward);
                SetJointMotor(leftKnee, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(leftKnee, 0, MathHelper.Pi * 0.8f);

                rightKnee = new RevoluteJoint(rightThigh, rightCalf, new Vector3(0, rightThigh.Position.Y - rightThigh.HalfHeight + 0.1f, pelvis.Height / 2 - THIGH_WIDTH / 2), Vector3.Forward);
                SetJointMotor(rightKnee, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(rightKnee, 0, MathHelper.Pi * 0.8f);

                leftAnkle = new RevoluteJoint(leftCalf, leftFoot, new Vector3(0, leftCalf.Position.Y - leftCalf.HalfHeight - 0.1f, -pelvis.Height / 2 + THIGH_WIDTH / 2), Vector3.Forward);
                SetJointMotor(leftAnkle, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(leftAnkle, -MathHelper.Pi / 2 - MathHelper.Pi / 4, -MathHelper.Pi / 2 + MathHelper.Pi / 6);

                rightAnkle = new RevoluteJoint(rightCalf, rightFoot, new Vector3(0, leftCalf.Position.Y - rightCalf.HalfHeight - 0.1f, pelvis.Height / 2 - THIGH_WIDTH / 2), Vector3.Forward);
                SetJointMotor(rightAnkle, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(rightAnkle, -MathHelper.Pi / 2 - MathHelper.Pi / 4, -MathHelper.Pi / 2 + MathHelper.Pi / 6);

                var leftFootJoint = new RevoluteJoint(leftFoot, leftFrontFoot, new Vector3(leftFoot.Position.X + leftFoot.HalfWidth, leftFoot.Position.Y - leftFoot.HalfHeight, -pelvis.Height / 2 + THIGH_WIDTH / 2), Vector3.Forward);
                SetJointLimit(leftFootJoint, -MathHelper.Pi / 4, -MathHelper.Pi / 4);

                var rightFootJoint = new RevoluteJoint(rightFoot, rightFrontFoot, new Vector3(rightFoot.Position.X + rightFoot.HalfWidth, rightFoot.Position.Y - rightFoot.HalfHeight, pelvis.Height / 2 - THIGH_WIDTH / 2), Vector3.Forward);
                SetJointLimit(rightFootJoint, -MathHelper.Pi / 4, -MathHelper.Pi / 4);

                lowerSpineJoint = new RevoluteJoint(torso, pelvis, new Vector3(0, pelvis.Position.Y - 0.05f, 0), Vector3.Forward);
                SetJointMotor(lowerSpineJoint, Owner.MOTOR_MAX_FORCE);
                SetJointLimit(lowerSpineJoint, MathHelper.Pi - MathHelper.Pi / 3, MathHelper.Pi + MathHelper.Pi / 6);

                #region Add all bodyparts to a list
                bodyparts = new List<Entity>();
                bodyparts.Add(torso);
                bodyparts.Add(pelvis);
                bodyparts.Add(leftUpperThigh);
                bodyparts.Add(rightUpperThigh);
                bodyparts.Add(leftThigh);
                bodyparts.Add(rightThigh);
                bodyparts.Add(leftCalf);
                bodyparts.Add(rightCalf);
                bodyparts.Add(leftFoot);
                bodyparts.Add(rightFoot);
                bodyparts.Add(leftFrontFoot);
                bodyparts.Add(rightFrontFoot);
                #endregion

                #region Add all joins to a list
                joints = new List<RevoluteJoint>();
                joints.Add(leftLateralHip);
                joints.Add(rightLateralHip);
                joints.Add(leftHip);
                joints.Add(rightHip);
                joints.Add(leftKnee);
                joints.Add(rightKnee);
                joints.Add(leftAnkle);
                joints.Add(rightAnkle);
                joints.Add(lowerSpineJoint);
                #endregion

                foreach (var joint in joints)
                {
                    joint.AngularJoint.SpringSettings.Stiffness = Owner.JOINT_STIFFNESS;
                }

                #region Add body parts to space
                Owner.Space.Add(pelvis);
                Owner.Space.Add(leftUpperThigh);
                Owner.Space.Add(rightUpperThigh);
                Owner.Space.Add(leftLateralHip);
                Owner.Space.Add(rightLateralHip);
                Owner.Space.Add(leftThigh);
                Owner.Space.Add(rightThigh);
                Owner.Space.Add(leftHip);
                Owner.Space.Add(rightHip);
                Owner.Space.Add(leftCalf);
                Owner.Space.Add(rightCalf);
                Owner.Space.Add(leftKnee);
                Owner.Space.Add(rightKnee);
                Owner.Space.Add(leftFoot);
                Owner.Space.Add(rightFoot);
                Owner.Space.Add(leftAnkle);
                Owner.Space.Add(rightAnkle);
                Owner.Space.Add(torso);
                Owner.Space.Add(lowerSpineJoint);
                Owner.Space.Add(leftFrontFoot);
                Owner.Space.Add(rightFrontFoot);
                Owner.Space.Add(leftFootJoint);
                Owner.Space.Add(rightFootJoint);
                #endregion

                #region Disable collisions between body parts
                CollisionRules.AddRule(pelvis, leftUpperThigh, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(pelvis, rightUpperThigh, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(leftUpperThigh, leftThigh, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(rightUpperThigh, rightThigh, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(pelvis, leftThigh, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(pelvis, rightThigh, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(leftThigh, leftCalf, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(rightThigh, rightCalf, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(leftCalf, leftFoot, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(rightCalf, rightFoot, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(torso, pelvis, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(leftFoot, leftFrontFoot, CollisionRule.NoBroadPhase);
                CollisionRules.AddRule(rightFoot, rightFrontFoot, CollisionRule.NoBroadPhase);
                #endregion
            }

            public override void Execute()
            {
                //Pass controls 
                Owner.Controls.SafeCopyToHost();

                switch (Owner.MOTOR_MODE) 
                {
                    case MyMotorMode.VELOCITY_MOTOR:
                        for (int i = 0; i < Owner.JOINTS; i++)
                        {
                            joints[i].Motor.Settings.VelocityMotor.GoalVelocity = Owner.Controls.Host[i];
                        }
                        break;
                    case MyMotorMode.SERVO:
                        for (int i = 0; i < Owner.JOINTS; i++)
                        {
                            joints[i].Motor.Settings.VelocityMotor.GoalVelocity = Owner.Controls.Host[i];
                        }
                        break;
                    case MyMotorMode.MUSCLE_PAIR:
                        for (int i = 0; i < Owner.JOINTS; i++)
                        {
                            SetMuscleSettings(joints[i], Owner.Controls.Host[2 * i], Owner.Controls.Host[2 * i + 1]);
                        }
                        break;
                }

                //Pushing the torso with outside force
                if (Owner.Push != null && Owner.Push.Count >= 3)
                {
                    Owner.Push.SafeCopyToHost();
                    torso.LinearVelocity = torso.LinearVelocity + new Vector3(Owner.Push.Host[0], Owner.Push.Host[1], Owner.Push.Host[2]);
                }

                //Update world simulation
                Owner.Space.Update();

                //Get feedback
                for (int i = 0; i < Owner.JOINTS; i++)
                {
                    Owner.Joints.Host[i] = GetJointRotation(joints[i]);
                }

                for (int i = 0; i < Owner.Joints.Count; i++)
                {
                    if (float.IsNaN(Owner.Joints.Host[i]))
                        Owner.Joints.Host[i] = 0;
                }

                Owner.TorsoRotation.Host[0] = torso.Orientation.X;
                Owner.TorsoRotation.Host[1] = torso.Orientation.Y;
                Owner.TorsoRotation.Host[2] = torso.Orientation.Z;

                Owner.FeetPressure.Host[0] = GetImpactForce(leftFoot);
                Owner.FeetPressure.Host[1] = GetImpactForce(leftFrontFoot);
                Owner.FeetPressure.Host[2] = GetImpactForce(rightFoot);
                Owner.FeetPressure.Host[3] = GetImpactForce(rightFrontFoot);

                //Find center of mass
                Vector3 center = GetCenterOfMass();
                Owner.CenterOfMass.Host[0] = center.X;
                Owner.CenterOfMass.Host[1] = center.Y;
                Owner.CenterOfMass.Host[2] = center.Z;

                //Find contact point
                Vector3 contact = GetContactPoint();
                Owner.CenterOfPressure.Host[0] = contact.X;
                Owner.CenterOfPressure.Host[1] = contact.Y;
                Owner.CenterOfPressure.Host[2] = contact.Z;

                //Joint position and axis
                for (int i = 0; i < Owner.JOINTS; i++)
                {
                    var anchorPosition = joints[i].BallSocketJoint.ConnectionA.Position + joints[i].BallSocketJoint.OffsetA;
                    Owner.JointPosition.Host[3 * i] = anchorPosition.X;
                    Owner.JointPosition.Host[3 * i + 1] = anchorPosition.Y;
                    Owner.JointPosition.Host[3 * i + 2] = anchorPosition.Z;

                    Owner.JointAxis.Host[3 * i] = joints[i].AngularJoint.WorldFreeAxisB.X;
                    Owner.JointAxis.Host[3 * i + 1] = joints[i].AngularJoint.WorldFreeAxisB.Y;
                    Owner.JointAxis.Host[3 * i + 2] = joints[i].AngularJoint.WorldFreeAxisB.Z;
                }

                Owner.TorsoPosition.Host[0] = torso.Position.X;
                Owner.TorsoPosition.Host[1] = torso.Position.Y;
                Owner.TorsoPosition.Host[2] = torso.Position.Z;

                Owner.Joints.SafeCopyToDevice();
                Owner.TorsoRotation.SafeCopyToDevice();
                Owner.FeetPressure.SafeCopyToDevice();
                Owner.CenterOfMass.SafeCopyToDevice();
                Owner.CenterOfPressure.SafeCopyToDevice();
                Owner.JointPosition.SafeCopyToDevice();
                Owner.JointAxis.SafeCopyToDevice();
                Owner.TorsoPosition.SafeCopyToDevice();

                Owner.Controls.CopyToMemoryBlock(Owner.ControlsCopy, 0, 0, Owner.ControlsCopy.Count);
            }

            private float GetJointRotation(RevoluteJoint joint)
            {
                return Vector3.Dot(Vector3.Normalize(joint.BallSocketJoint.OffsetA), Vector3.Normalize(joint.BallSocketJoint.OffsetB));
            }

            private float GetImpactForce(Entity entity)
            {
                float impactForce = 0.0f;
                for (int i = 0; i < entity.CollisionInformation.Pairs.Count; i++)
                {
                    for (int j = 0; j < entity.CollisionInformation.Pairs[i].Contacts.Count; j++)
                    {
                        impactForce += entity.CollisionInformation.Pairs[i].Contacts[j].NormalImpulse;
                    }
                }
                return impactForce;
            }

            private void SetMuscleSettings(RevoluteJoint joint, float flexorForce, float extensorForce)
            {
                flexorForce = Math.Max(flexorForce, 0);
                extensorForce = Math.Max(extensorForce, 0);

                float totalVelocity = (float) Math.Pow(VELOCITY_MULTIPLIER * (extensorForce - flexorForce), 3);
                joint.Motor.Settings.VelocityMotor.GoalVelocity = totalVelocity;
                joint.Motor.Settings.MaximumForce = (float)Math.Pow(FORCE_MULTIPLIER * (extensorForce + flexorForce), 3);
            }

            private void SetJointMotor(RevoluteJoint joint, float maxForce)
            {
                joint.Motor.IsActive = true;
                joint.Motor.Settings.Mode = Owner.MOTOR_MODE == MyMotorMode.SERVO ? MotorMode.Servomechanism : MotorMode.VelocityMotor;
                joint.Motor.Settings.MaximumForce = maxForce;
            }

            private void SetJointLimit(RevoluteJoint joint, float min, float max)
            {
                joint.Limit.IsActive = true;
                joint.Limit.MinimumAngle = min;
                joint.Limit.MaximumAngle = max;
            }

            private Vector3 GetCenterOfMass()
            {
                Vector3 center = new Vector3(0, 0, 0);
                float totalMass = torso.Mass + pelvis.Mass + 2 * leftThigh.Mass + 2 * leftCalf.Mass + 2 * leftFoot.Mass + 2 * leftFrontFoot.Mass;

                for (int i = 0; i < bodyparts.Count; i++)
                {
                    center += bodyparts[i].Position * bodyparts[i].Mass;
                }

                center = center / totalMass;
                return center;
            }

            private Vector3 GetContactPoint()
            {
                Vector3 contact = new Vector3(0, 0, 0);
                float totalPressure = 0.0f;
                for (int i = 0; i < Owner.FeetPressure.Count; i++)
                {
                    totalPressure += Owner.FeetPressure.Host[i];
                }
                if (totalPressure > 0)
                {
                    contact += leftFoot.Position * Owner.FeetPressure.Host[0];
                    contact += leftFrontFoot.Position * Owner.FeetPressure.Host[1];
                    contact += rightFoot.Position * Owner.FeetPressure.Host[2];
                    contact += rightFrontFoot.Position * Owner.FeetPressure.Host[3];

                    contact = contact / totalPressure;
                }
                return contact;
            }
        }
    }
}
