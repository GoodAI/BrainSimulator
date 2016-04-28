using World.GameActors;
﻿using World.GameActors.Tiles;
using World.ToyWorldCore;

namespace World.GameActions
{
    public abstract class GameAction
    {
        protected GameActor m_sender { get; set; }

        protected GameAction(GameActor sender)
        {
            m_sender = sender;
        }

        public virtual void Resolve(GameActor target, IAtlas atlas) { }
    }

    public class ToUsePickaxe : GameAction
    {
        public float Damage { get; set; }

        public ToUsePickaxe(GameActor sender) : base(sender) { }
    }

    public class PickUp : GameAction
    {
        public PickUp(GameActor sender) : base(sender) { }

        public override void Resolve(GameActor target, IAtlas atlas)
        {
            ICanPick picker = m_sender as ICanPick;
            IPickable pickItem = target as IPickable;

            if (picker != null && pickItem != null)
                picker.AddToInventory(pickItem);

            atlas.Remove(target);
        }
    }
}