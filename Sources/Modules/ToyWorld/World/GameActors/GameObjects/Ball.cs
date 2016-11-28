using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles;
using World.Physics;
using World.ToyWorldCore;

namespace World.GameActors.GameObjects
{
    class Ball : Character, IPickableGameActor
    {
        public Ball(
            string tilesetName,
            int tileId, string name,
            Vector2 position,
            Vector2 size,
            float direction)
            : base(
            tilesetName,
            tileId,
            name,
            position,
            size,
            direction,
            typeof(CircleShape))
        {
        }

        public void PickUp(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable = null)
        {
            if (gameAction is PickUp || gameAction is LayDown)
            {
                gameAction.Resolve(new GameActorPosition(this, position, LayerType.Object), atlas, tilesetTable);
            }
        }
    }
}
