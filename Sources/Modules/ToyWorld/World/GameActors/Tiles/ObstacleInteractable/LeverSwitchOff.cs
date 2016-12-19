using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitchOff : DynamicTile, ISwitcherGameActor, IInteractableGameActor
    {
        public ISwitchableGameActor Switchable { get; set; }

        public LeverSwitchOff(Vector2I position) : base(position) { } 

 		public LeverSwitchOff(Vector2I position, int textureId) : base(position, textureId) { }

        public LeverSwitchOff(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            var leverOn = new LeverSwitchOn(Position);
            atlas.ReplaceWith(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable), leverOn);
            if (Switchable == null) return;
            leverOn.Switchable = Switchable.Switch(null, atlas);
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity, LayerType.ObstacleInteractable), atlas);
        }
    }
}