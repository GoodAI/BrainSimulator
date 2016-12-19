using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors.Tiles.ObstacleInteractable;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class KeyDoorOpened : DynamicTile, ISwitchableGameActor
    {
        public KeyDoorOpened(Vector2I position) : base(position) { } 

 		public KeyDoorOpened(Vector2I position, int textureId) : base(position, textureId) { }

        public KeyDoorOpened(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public ISwitchableGameActor Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            KeyDoorClosed closedDoor = new KeyDoorClosed(Position);
            bool added = atlas.Add(new GameActorPosition(closedDoor, (Vector2)Position, LayerType.ObstacleInteractable), true);
            if (!added) return this;
            atlas.Remove(new GameActorPosition(this, (Vector2)Position, LayerType.OnGroundInteractable));
            return closedDoor;
        }

        public ISwitchableGameActor SwitchOn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            return this;
        }

        public ISwitchableGameActor SwitchOff(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            return Switch(gameActorPosition, atlas);
        }
    }
}