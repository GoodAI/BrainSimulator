using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;

namespace World.GameActors.Tiles.ObstacleInteractable
{
    public class LeverSwitchOn : DynamicTile, ISwitcherGameActor, IInteractableGameActor
    {
        public ISwitchableGameActor Switchable { get; set; }

        public LeverSwitchOn(Vector2I position) : base(position) { } 

        public LeverSwitchOn(Vector2I position, int textureId) : base(position, textureId) { }

        public LeverSwitchOn(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public void Switch(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            var leverOff = new LeverSwitchOff(Position);
            atlas.ReplaceWith(new GameActorPosition(this, (Vector2)Position, LayerType.ObstacleInteractable), leverOff);
            if (Switchable == null) return;
            leverOff.Switchable = Switchable.Switch(null, atlas) as ISwitchableGameActor;
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            gameAction.Resolve(new GameActorPosition(this, Vector2.PositiveInfinity, LayerType.ObstacleInteractable), atlas);
        }
    }
}