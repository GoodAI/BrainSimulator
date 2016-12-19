using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles.ObstacleInteractable;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class SimpleDoorOpened : StaticTile, IInteractableGameActor
    {
        public SimpleDoorOpened()
        {
        }

        public SimpleDoorOpened(string textureName)
            : base(textureName)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            var doorClosed = new SimpleDoorClosed();
            bool added = atlas.Add(new GameActorPosition(doorClosed, position, LayerType.ObstacleInteractable), true);
            if (added)
            {
                atlas.Remove(new GameActorPosition(this, position, LayerType.OnGroundInteractable));
            }
        }
    }
}