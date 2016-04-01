namespace GoodAI.ToyWorld.Control
{
    // TODO : delete this interface
    /// <summary>
    /// Deprecated
    /// </summary>
    public interface IAvatarAction
    {
        /// <summary>
        /// 
        /// Value is clamped to (-1,1). Negative values mean move backwards, positive are for forward movement.
        /// </summary>
        float Acceleration { set; }

        /// <summary>
        /// 
        /// Value is clamped to (-1,1). Negative values mean rotate left, positive are for rotation to the right.
        /// </summary>
        float Rotation { set; }


        /// <summary>
        /// 
        /// </summary>
        bool Use { set; }

        /// <summary>
        /// 
        /// </summary>
        bool Interact { set; }

        /// <summary>
        /// 
        /// </summary>
        bool Pick { set; }
    }
}
