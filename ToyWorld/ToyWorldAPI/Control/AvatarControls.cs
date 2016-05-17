using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    ///
    /// </summary>
    public interface IAvatarControls
    {
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        AvatarAction<float> DesiredForwardSpeed { get; }

        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move left, positive are for right movement.
        /// </summary>
        AvatarAction<float> DesiredRightSpeed { get; }

        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        AvatarAction<float> DesiredRotation { get; }

        /// <summary>
        /// To interact with object in front.
        /// </summary>
        AvatarAction<bool> Interact { get; }

        /// <summary>
        /// To use tool in hand / punch.
        /// </summary>
        AvatarAction<bool> Use { get; }

        /// <summary>
        /// Pick up or put down tool in hand.
        /// </summary>
        AvatarAction<bool> PickUp { get; }

        /// <summary>
        /// Set Fof position
        /// </summary>
        AvatarAction<PointF> Fof { get; }

        /// <summary>
        /// Rewrites actions from this list with actions from parameter with lower priority value.
        /// </summary>
        /// <param name="actions"></param>
        void Update(IAvatarControls actions);

        /// <summary>
        /// Returns actions as a float array
        /// </summary>
        /// <returns></returns>
        float[] ToArray();
    }

    /// <summary>
    ///
    /// </summary>
    public struct AvatarControls : IAvatarControls
    {
        private AvatarAction<float> m_desiredForwardSpeed;
        private AvatarAction<float> m_desiredRightSpeed;
        private AvatarAction<float> m_desiredRotation;
        private AvatarAction<bool> m_interact;
        private AvatarAction<bool> m_use;
        private AvatarAction<bool> m_pickUp;
        private AvatarAction<PointF> m_fof;

        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        public AvatarAction<float> DesiredForwardSpeed { get { return m_desiredForwardSpeed; } set { m_desiredForwardSpeed += value; } }

        public AvatarAction<float> DesiredRightSpeed { get { return m_desiredRightSpeed; } set { m_desiredRightSpeed += value; } }

        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        public AvatarAction<float> DesiredRotation { get { return m_desiredRotation; } set { m_desiredRotation += value; } }

        /// <summary>
        /// To interact with object in front.
        /// </summary>
        public AvatarAction<bool> Interact { get { return m_interact; } set { m_interact += value; } }

        /// <summary>
        /// To use tool in hand / punch.
        /// </summary>
        public AvatarAction<bool> Use { get { return m_use; } set { m_use += value; } }

        /// <summary>
        /// Pick up or put down tool in hand.
        /// </summary>
        public AvatarAction<bool> PickUp { get { return m_pickUp; } set { m_pickUp += value; } }

        /// <summary>
        /// Set Fof position
        /// </summary>
        public AvatarAction<PointF> Fof { get { return m_fof; } set { m_fof += value; } }


        ///  <summary>
        ///
        ///  </summary>
        ///  <param name="priority"></param>
        /// <param name="desiredForwardSpeed"></param>
        /// <param name="desiredRightSpeed"></param>
        /// <param name="desiredRotation"></param>
        ///  <param name="interact"></param>
        ///  <param name="use"></param>
        ///  <param name="pickUp"></param>
        ///  <param name="fof"></param>
        public AvatarControls(
            int priority,
            float desiredForwardSpeed = 0f,
            float desiredRightSpeed = 0f,
            float desiredRotation = 0f,
            bool interact = false,
            bool use = false,
            bool pickUp = false,
            PointF fof = default(PointF)
            )
            : this()
        {
            m_desiredForwardSpeed = new AvatarAction<float>(desiredForwardSpeed, priority);
            m_desiredRightSpeed = new AvatarAction<float>(desiredRightSpeed, priority);
            m_desiredRotation = new AvatarAction<float>(desiredRotation, priority);
            m_interact = new AvatarAction<bool>(interact, priority);
            m_use = new AvatarAction<bool>(use, priority);
            m_pickUp = new AvatarAction<bool>(pickUp, priority);
            m_fof = new AvatarAction<PointF>(fof, priority);
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="other"></param>
        public AvatarControls(IAvatarControls other)
            : this()
        {
            Update(other);
        }

        /// <summary>
        /// Rewrites actions from this AvatarControls with actions from given AvatarControls with lower priority value.
        /// </summary>
        /// <param name="actions"></param>
        public void Update(IAvatarControls actions)
        {
            if (actions == null)
                return;

            DesiredForwardSpeed = actions.DesiredForwardSpeed;
            DesiredRightSpeed = actions.DesiredRightSpeed;
            DesiredRotation = actions.DesiredRotation;
            Interact = actions.Interact;
            Use = actions.Use;
            PickUp = actions.PickUp;
            Fof = actions.Fof;
        }

        public float[] ToArray()
        {
            List<float> result = new List<float>
            {
                DesiredForwardSpeed,
                DesiredRightSpeed,
                DesiredRotation,
                Interact.Value ? 1 : 0,
                Use.Value ? 1 : 0,
                PickUp.Value ? 1 : 0,
                Fof.Value.X,
                Fof.Value.Y
            };

            return result.ToArray();
        }
    }
}
