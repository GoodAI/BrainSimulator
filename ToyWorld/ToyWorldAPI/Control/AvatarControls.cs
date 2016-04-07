using System.Collections.Generic;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public class AvatarControls
    {
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        public AvatarAction<float> DesiredSpeed { get; set; }
        /// <summary>
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        public AvatarAction<float> DesiredRotation { get; set; }
        /// <summary>
        /// To interact with object in front.
        /// </summary>
        public AvatarAction<bool> Interact { get; set; }
        /// <summary>
        /// To use tool in hand / punch.
        /// </summary>
        public AvatarAction<bool> Use { get; set; }
        /// <summary>
        /// Pick up or put down tool in hand.
        /// </summary>
        public AvatarAction<bool> PickUp { get; set; }

        public AvatarControls()
        {
            DesiredSpeed = new AvatarAction<float>(0, 0);
            DesiredRotation = new AvatarAction<float>(0, 0);
            Interact = new AvatarAction<bool>(false, 0);
            Use = new AvatarAction<bool>(false, 0);
            PickUp = new AvatarAction<bool>(false, 0);
        } 

        /// <summary>
        /// Rewrites actions from this list with actions from parameter with lower priority value.
        /// </summary>
        /// <param name="actions"></param>
        public void Update(AvatarControls actions)
        {
            if (DesiredSpeed.Priority < actions.DesiredSpeed.Priority)
            {
                DesiredSpeed = actions.DesiredSpeed;
            }
            if (DesiredRotation.Priority < actions.DesiredRotation.Priority)
            {
                DesiredRotation = actions.DesiredRotation;
            }
            if (Interact.Priority < actions.Interact.Priority)
            {
                Interact = actions.Interact;
            }
            if (Use.Priority < actions.Use.Priority)
            {
                Use = actions.Use;
            }
            if (PickUp.Priority < actions.PickUp.Priority)
            {
                PickUp = actions.PickUp;
            }
        }
    }
}
