using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.Tiles.OnGroundInteractable;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class KeyDoorClosed : DynamicTile, ISwitchableGameActor
    {
        public KeyDoorClosed(Vector2I position) : base(position) { } 

 		public KeyDoorClosed(Vector2I position, int textureId) : base(position, textureId) { }

        public KeyDoorClosed(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            KeyDoorOpened openedDoor = new KeyDoorOpened(Position);
            bool added = atlas.Add(new GameActorPosition(openedDoor, (Vector2)Position, LayerType.OnGroundInteractable));
            if (!added) return this;
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable));
            return openedDoor;
        }

        public ISwitchableGameActor SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            return Switch(gameActorPosition, atlas);
        }

        public ISwitchableGameActor SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            return this;
        }
    }
}