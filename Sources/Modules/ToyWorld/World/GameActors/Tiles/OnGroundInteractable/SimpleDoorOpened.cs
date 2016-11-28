using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class SimpleDoorOpened : StaticTile, IInteractableGameActor
    {
        public SimpleDoorOpened(ITilesetTable tilesetTable)
            : base(tilesetTable)
        {
        }

        public SimpleDoorOpened(int tileType)
            : base(tileType)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable)
        {
            var doorClosed = new SimpleDoorClosed(tilesetTable);
            bool added = atlas.Add(new GameActorPosition(doorClosed, position, LayerType.ObstacleInteractable), true);
            if (added)
            {
                atlas.Remove(new GameActorPosition(this, position, LayerType.OnGroundInteractable));
            }
        }
    }
}