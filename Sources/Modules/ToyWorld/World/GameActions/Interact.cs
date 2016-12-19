using World.Atlas;
using World.GameActors;
using World.GameActors.Tiles.ObstacleInteractable;

namespace World.GameActions
{
    class Interact : GameAction
    {
        public Interact(GameActor sender) : base(sender)
        {
        }

        public override void Resolve(GameActorPosition target, IAtlas atlas)
        {
            if (target.Actor is Apple || target.Actor is Pear)
            {
                atlas.Remove(target);
            }

            var switcher = target.Actor as ISwitcherGameActor;

            if (switcher != null)
            {
                switcher.Switch(target, atlas);
            }
        }
    }
}
