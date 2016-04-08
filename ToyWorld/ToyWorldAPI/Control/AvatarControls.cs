using System.Collections.Generic;

namespace GoodAI.ToyWorld.Control
{
    public interface IAvatarControls
    {
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        AvatarAction<float> DesiredSpeed { get; }

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
        /// Rewrites actions from this list with actions from parameter with lower priority value.
        /// </summary>
        /// <param name="actions"></param>
        void Update(IAvatarControls actions);
    }

    /// <summary>
    /// 
    /// </summary>
    public class AvatarControls : IAvatarControls
    {
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        public AvatarAction<float> DesiredSpeed { get; private set; }
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        public AvatarAction<float> DesiredRotation { get; private set; }
        /// <summary>
        /// To interact with object in front.
        /// </summary>
        public AvatarAction<bool> Interact { get; private set; }
        /// <summary>
        /// To use tool in hand / punch.
        /// </summary>
        public AvatarAction<bool> Use { get; private set; }
        /// <summary>
        /// Pick up or put down tool in hand.
        /// </summary>
        public AvatarAction<bool> PickUp { get; private set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="desiredSpeed"></param>
        /// <param name="desiredRotation"></param>
        /// <param name="interact"></param>
        /// <param name="use"></param>
        /// <param name="pickUp"></param>
        public AvatarControls(
            AvatarAction<float> desiredSpeed,
            AvatarAction<float> desiredRotation,
            AvatarAction<bool> interact,
            AvatarAction<bool> use,
            AvatarAction<bool> pickUp
            )
        {
            DesiredSpeed = desiredSpeed;
            DesiredRotation = desiredRotation;
            Interact = interact;
            Use = use;
            PickUp = pickUp;
        }

        public AvatarControls(
            int priority,
            float desiredSpeed = 0f,
            float desiredRotation = 0f,
            bool interact = false,
            bool use = false,
            bool pickUp = false
            )
        {
            DesiredSpeed = new AvatarAction<float>(desiredSpeed, priority);
            DesiredRotation = new AvatarAction<float>(desiredRotation, priority);
            Interact = new AvatarAction<bool>(interact, priority);
            Use = new AvatarAction<bool>(use, priority);
            PickUp = new AvatarAction<bool>(pickUp, priority);
        }

        /// <summary>
        /// Rewrites actions from this AvatarControls with actions from given AvatarControls with lower priority value.
        /// </summary>
        /// <param name="actions"></param>
        public void Update(IAvatarControls actions)
        {
            if (DesiredSpeed.Priority > actions.DesiredSpeed.Priority)
            {
                DesiredSpeed = actions.DesiredSpeed;
            }
            if (DesiredRotation.Priority > actions.DesiredRotation.Priority)
            {
                DesiredRotation = actions.DesiredRotation;
            }
            if (Interact.Priority > actions.Interact.Priority)
            {
                Interact = actions.Interact;
            }
            if (Use.Priority > actions.Use.Priority)
            {
                Use = actions.Use;
            }
            if (PickUp.Priority > actions.PickUp.Priority)
            {
                PickUp = actions.PickUp;
            }
        }
    }
}
