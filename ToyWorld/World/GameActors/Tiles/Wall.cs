using System;
using World.GameActions;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     Wall can be transformed to DamagedWall if pickaxe is used
    /// </summary>
    public class Wall : StaticTile, IInteractable
    {

        public Wall(int tileType) : base(tileType)
        {
        }

        public Tile ApplyGameAction(GameAction gameAction, TilesetTable tilesetTable)
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
                    return new DestroyedWall(gameAction.TilesetTable);
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
        private DamagedWall(TilesetTable tilesetTable) : base(tilesetTable)
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

        public float Health { get; private set; }

        public Tile ApplyGameAction(GameAction gameAction, TilesetTable tilesetTable)
        {
            if (gameAction is ToUsePickaxe)
            {
                var usePickaxe = (ToUsePickaxe) gameAction;
                Health -= usePickaxe.Damage;
            }
            if (Health <= 0f)
            {
                return new DestroyedWall(gameAction.TilesetTable);
            }
            return this;
        }
    }

    /// <summary>
    /// </summary>
    public class DestroyedWall : StaticTile
    {
        public DestroyedWall(int tileType) : base(tileType)
        {
        }

        public DestroyedWall(TilesetTable tilesetTable) : base(tilesetTable)
        {
        }
    }
}