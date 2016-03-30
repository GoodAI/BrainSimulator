using System;
using World.GameActions;
using World.Tiles;

namespace World.GameActors.Tiles
{
    /// <summary>
    ///     Wall can be transformed to DamagedWall if pickaxe is used
    /// </summary>
    public class Wall : StaticTile, INteractable
    {
        public Wall(TileSetTableParser tileSetTableParser)
        {
            TileType = tileSetTableParser.TileNumber("Wall");
        }

        public Tile ApplyGameAction(GameAction gameAction)
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
                    return new DestroyedWall();
                }
                return new DamagedWall((gameAction as ToUsePickaxe).Damage);
            }
            return this;
        }
    }

    /// <summary>
    ///     DamagedWall has health from (0,1) excl. If health leq 0, it is replaced by DestroyedWall.
    ///     Only way how to make damage is to use pickaxe.
    /// </summary>
    public class DamagedWall : DynamicTile, INteractable
    {
        private DamagedWall()
        {
            Health = 1f;
        }

        public DamagedWall(float damage) : this()
        {
            Health -= damage;
        }

        public float Health { get; private set; }

        public Tile ApplyGameAction(GameAction gameAction)
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
            return this;
        }
    }

    /// <summary>
    /// </summary>
    public class DestroyedWall : StaticTile
    {
        public Tile TransformTo(GameAction gameAction)
        {
            return this;
        }
    }
}