using System;
using World.GameActions;

namespace World.Tiles
{
    /// <summary>
    /// Wall can be transformed to DamagedWall if pickaxe is used
    /// </summary>
    public class Wall : StaticTile, Interactable
    {
        public AbstractTile ApplyGameAction(GameAction gameAction)
        {
            if (gameAction is ToUsePickaxe)
            {
                var toUsePickaxe = ((ToUsePickaxe) gameAction);
                if (Math.Abs(toUsePickaxe.Damage) < 0.00001f)
                {
                    return null;
                }
                if (toUsePickaxe.Damage >= 1.0f)
                {
                    return new DestroyedWall();
                }
                return new DamagedWall((gameAction as ToUsePickaxe).Damage);
            }
            return null;
        }

        public override string Name
        {
            get { return "Wall"; }
        }
    }

    /// <summary>
    /// DamagedWall has health from (0,1) excl. If health leq 0, it is replaced by DestroyedWall.
    /// Only way how to make damage is to use pickaxe.
    /// </summary>
    public class DamagedWall : DynamicTile, Interactable
    {
        public float Health { get; private set; }

        private DamagedWall()
        {
            Health = 1f;
        }

        public DamagedWall(float damage) : this()
        {
            Health -= damage;
        }

        public AbstractTile ApplyGameAction(GameAction gameAction)
        {
            if (gameAction is ToUsePickaxe)
            {
                var usePickaxe = (ToUsePickaxe) gameAction;
                Health -= usePickaxe.Damage;
            }
            if (Health <= 0f)
            {
                return new DestroyedWall();
            }
            return null;
        }

        public override string Name
        {
            get { return "DamagedWall"; }
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class DestroyedWall : StaticTile
    {
        public AbstractTile TransformTo(GameAction gameAction)
        {
            return null;
        }

        public override string Name
        {
            get { return "DestroyedWall"; }
        }
    }
}

