using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles.OnGroundInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class SimpleDoorClosed : StaticTile, IInteractable
    {
        public SimpleDoorClosed(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }

        public SimpleDoorClosed(int tileType) : base(tileType)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            var doorOpened = new SimpleDoorOpened(tilesetTable);
            bool added = atlas.Add(new GameActorPosition(doorOpened, position, LayerType.OnGroundInteractable));
            if (added)
            {
                atlas.Remove(new GameActorPosition(this, position, LayerType.ObstacleInteractable));
            }
        }
    }
}