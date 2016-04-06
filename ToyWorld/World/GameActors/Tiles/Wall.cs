using System;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     Wall can be transformed to DamagedWall if pickaxe is used
    /// </summary>
    public class Wall : StaticTile, IInteractable
    {
        public Wall(ITilesetTable tilesetTable)
            : base(tilesetTable)
        {
        }

        public Tile ApplyGameAction(Atlas atlas, GameAction gameAction, TilesetTable tilesetTable)
        {
            if (gameAction is ToUsePickaxe)
            {
                var toUsePickaxe = (ToUsePickaxe) gameAction;
                if (Math.Abs(toUsePickaxe.Damage) < 0.00001f)
                {
                    return this;
                }
                if (toUsePickaxe.Damage >= 1.0f)
                {
                    return new DestroyedWall(tilesetTable);
                }
                return new DamagedWall((gameAction as ToUsePickaxe), tilesetTable);
            }
            return this;
        }
    }

    /// <summary>
    ///     DamagedWall has health from (0,1) excl. If health leq 0, it is replaced by DestroyedWall.
    ///     Only way how to make damage is to use pickaxe.
    /// </summary>
    public class DamagedWall : DynamicTile, IInteractable
    {
        public float Health { get; private set; }

        private DamagedWall(ITilesetTable tilesetTable) : base(tilesetTable)
        {
            Health = 1f;
        }

        public DamagedWall(float damage, TilesetTable tilesetTable)
            : this(tilesetTable)
        {
            Health -= damage;
        }

        public DamagedWall(ToUsePickaxe toUsePickaxe, TilesetTable tilesetTable)
            : this(toUsePickaxe.Damage, tilesetTable)
        {
        }



        public Tile ApplyGameAction(Atlas atlas, GameAction gameAction, TilesetTable tilesetTable)
        {
            if (gameAction is ToUsePickaxe)
            {
                var usePickaxe = (ToUsePickaxe) gameAction;
                Health -= usePickaxe.Damage;
            }
            if (Health <= 0f)
            {
                return new DestroyedWall(tilesetTable);
            }
            return this;
        }
    }

    /// <summary>
    /// </summary>
    public class DestroyedWall : StaticTile
    {
        public DestroyedWall(ITilesetTable tilesetTable) : base(tilesetTable)
        {
        }
    }
}