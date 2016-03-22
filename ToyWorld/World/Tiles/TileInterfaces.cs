using System;

namespace World.Tiles
{
    public interface ITransformable
    {
        AbstractTile TransformTo(GameAction gameAction);
    }
    /// <summary>
    /// Objects which interacts with Pickaxe should implement this
    /// </summary>
    public interface IActionReciever
    {
        void RecieveAction(float damage);
    }
}