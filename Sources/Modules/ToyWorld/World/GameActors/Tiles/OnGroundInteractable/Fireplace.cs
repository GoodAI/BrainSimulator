using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;

namespace World.GameActors.Tiles.OnGroundInteractable
{
    public class Fireplace : DynamicTile, IInteractableGameActor, ICombustibleGameActor
    {
        public Fireplace(Vector2I position) : base(position) {}

        public Fireplace(Vector2I position, int textureId) : base(position, textureId) { }

        public Fireplace(Vector2I position, string textureName) : base(position, textureName)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position)
        {
            var interact = gameAction as Interact;

            if (interact != null)
            {
                Ignite(atlas);
            }
        }

        public void Burn(GameActorPosition gameActorPosition, IAtlas atlas)
        {
            Ignite(atlas);
        }

        private void Ignite(IAtlas atlas)
        {
            var fireplaceBurning = new FireplaceBurning(Position);
            atlas.ReplaceWith(ThisGameActorPosition(LayerType.OnGroundInteractable), fireplaceBurning);
            atlas.RegisterToAutoupdate(fireplaceBurning);
        }
    }
}