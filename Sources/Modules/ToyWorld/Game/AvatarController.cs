using System;
using GoodAI.ToyWorld.Control;
using GoodAI.ToyWorldAPI;
using VRageMath;
using World.GameActors.GameObjects;

namespace Game
{
    public class AvatarController : IAvatarController
    {
        private readonly IAvatar m_avatar;
        private AvatarControls m_avatarControls;

        public string InMessage { get; set; }

        public string OutMessage
        {
            get { return m_avatar.OutMessage; }
            set { m_avatar.OutMessage = value; }
        }

        public AvatarController(IAvatar avatar)
        {
            m_avatar = avatar;
            m_avatarControls = new AvatarControls(int.MaxValue);
        }

        public void SetActions(IAvatarControls actions)
        {
            m_avatarControls.Update(actions);
            SetAvatarActionsControllable();
        }

        public IAvatarControls GetActions()
        {
            float desiredForwardSpeed = DesiredForwardSpeed();
            float desiredRightSpeed = DesiredRightSpeed();
            float desiredLeftRotation = m_avatar.DesiredLeftRotation;
            bool interact = m_avatar.Interact;
            bool pickUp = m_avatar.PickUp;
            bool use = m_avatar.UseTool;
            var fof = m_avatar.Fof;

            AvatarControls realAvatarControls = new AvatarControls(int.MaxValue, 
                desiredForwardSpeed, desiredRightSpeed, desiredLeftRotation, interact, use, pickUp, fof);
            //return m_avatarControls;
            return realAvatarControls;
        }

        public IStats GetStats()
        {
            throw new NotImplementedException();
        }

        public string GetComment()
        {
            throw new NotImplementedException();
        }

        public void ResetControls()
        {
            m_avatarControls = new AvatarControls(int.MaxValue);
            m_avatar.ResetControls();
        }

        private void SetAvatarActionsControllable()
        {
            float fSpeed = m_avatarControls.DesiredForwardSpeed;
            float rSpeed = m_avatarControls.DesiredRightSpeed;

            // diagonal strafing speed should not be greater than 1
            // speed must be between [0,1]

            var jointSpeed = JointSpeed(fSpeed, rSpeed);
            m_avatar.DesiredSpeed = jointSpeed;

            float jointDirection =
                JointDirection(fSpeed, rSpeed);
            m_avatar.Direction = jointDirection;
            m_avatar.DesiredLeftRotation = m_avatarControls.DesiredLeftRotation;
            m_avatar.Interact = m_avatarControls.Interact;
            m_avatar.PickUp = m_avatarControls.PickUp;
            m_avatar.UseTool = m_avatarControls.Use;
            m_avatar.Fof = m_avatarControls.Fof;
        }

        private float JointDirection(float fSpeed, float rSpeed)
        {
            return MathHelper.WrapAngle(m_avatar.Rotation
                                        + (float)Math.Atan2(fSpeed, rSpeed)
                                        - MathHelper.PiOver2); // Our zero angle is the up direction (instead of right)
        }

        /// <summary>
        /// Diagonal strafing speed should not be greater than 1.
        /// Speed must be between [0,1].
        /// </summary>
        /// <param name="fSpeed">[-1,1]</param>
        /// <param name="rSpeed">[-1,1]</param>
        /// <returns></returns>
        private static float JointSpeed(float fSpeed, float rSpeed)
        {
            // WolframAlpha.com: Plot[Sqrt((a^2+b^2)/(a+b)), {a,0,1}, {b,0,1}]
            //float jointSpeed = (float) Math.Sqrt((fSpeed*fSpeed + rSpeed*rSpeed)/(Math.Sqrt(fSpeed) + Math.Sqrt(rSpeed)));

            // WolframAlpha.com: Plot[Max(a,b), {a,0,1}, {b,0,1}]
            float jointSpeed = Math.Max(Math.Abs(fSpeed), Math.Abs(rSpeed));
            return jointSpeed;
        }

        private float DesiredForwardSpeed()
        {
            var a1 = Math.Cos(m_avatar.Rotation);
            var a2 = Math.Sin(m_avatar.Rotation);

            var b1 = Math.Cos(m_avatar.Direction) * m_avatar.DesiredSpeed;
            var b2 = Math.Sin(m_avatar.Direction) * m_avatar.DesiredSpeed;

            var lapt = a1 * a1 + a2 * a2;
            var atb = a1 * b1 + a2 * b2;

            return (float) (atb / lapt);
        }

        private float DesiredRightSpeed()
        {
            var b1 = Math.Cos(m_avatar.Direction) * m_avatar.DesiredSpeed;
            var b2 = Math.Sin(m_avatar.Direction) * m_avatar.DesiredSpeed;

            var lb = Math.Sqrt(b1 * b1 + b2 * b2);

            var sinTh = Math.Sin(m_avatar.Rotation - m_avatar.Direction);

            return (float) (sinTh * lb);
        }
    }
}
