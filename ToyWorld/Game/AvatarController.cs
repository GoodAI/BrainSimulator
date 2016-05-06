using System;
using System.Collections.Generic;
using GoodAI.ToyWorld.Control;
using VRageMath;
using World.GameActors.GameObjects;

namespace Game
{
    public class AvatarController : IAvatarController
    {
        private readonly IAvatar m_avatar;
        private AvatarControls m_avatarControls;

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

            // WolframAlpha.com: Plot[Sqrt((a^2+b^2)/(a+b)), {a,0,1}, {b,0,1}]
            //float jointSpeed = (float) Math.Sqrt((fSpeed*fSpeed + rSpeed*rSpeed)/(Math.Sqrt(fSpeed) + Math.Sqrt(rSpeed)));

            // WolframAlpha.com: Plot[Max(a,b), {a,0,1}, {b,0,1}]
            float jointSpeed = (float) Math.Max(Math.Abs(fSpeed), Math.Abs(rSpeed));

            m_avatar.DesiredSpeed = jointSpeed;
            m_avatar.Direction =
                MathHelper.WrapAngle(
                    m_avatar.Rotation +
                    (float) Math.Atan2(
                        m_avatarControls.DesiredRightSpeed,
                        m_avatarControls.DesiredForwardSpeed));
            m_avatar.DesiredRotation = m_avatarControls.DesiredRotation;
            m_avatar.Interact = m_avatarControls.Interact;
            m_avatar.PickUp = m_avatarControls.PickUp;
            m_avatar.DesiredRotation = m_avatarControls.DesiredRotation;
            m_avatar.Use = m_avatarControls.Use;
            m_avatar.Fof = m_avatarControls.Fof;
        }
    }
}
