using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors.Tiles.OnGroundInteractable;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class SimpleDoorClosed : StaticTile, IInteractableGameActor
    {
        public SimpleDoorClosed() : base (){ } 

 		public SimpleDoorClosed(int textureId) : base(textureId) { }

        public SimpleDoorClosed(string textureName)
            : base(textureName)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            var doorOpened = new SimpleDoorOpened();
            bool added = atlas.Add(new GameActorPosition(doorOpened, position, LayerType.OnGroundInteractable));
            if (added)
            {
                atlas.Remove(new GameActorPosition(this, position, LayerType.ObstacleInteractable));
            }
        }
    }
}