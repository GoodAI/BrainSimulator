using System;
using VRageMath;
using World.GameActions;
using World.ToyWorldCore;

namespace World.GameActors.Tiles.Obstacle
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

        public Wall(int tileType)
            : base(tileType)
        {
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable = null)
        {
            if (!(gameAction is UsePickaxe))
                return;

            UsePickaxe usePickaxe = (UsePickaxe)gameAction;
            if (Math.Abs(usePickaxe.Damage) < 0.00001f)
                return;

            if (usePickaxe.Damage >= 1.0f)
            {
                atlas.ReplaceWith(new GameActorPosition(this, position), new DestroyedWall(tilesetTable));
                return;
            }
            atlas.ReplaceWith(new GameActorPosition(this, position), new DamagedWall(usePickaxe.Damage, tilesetTable, Vector2I.Zero));
        }
    }

    /// <summary>
    ///     DamagedWall has health from (0,1) excl. If health leq 0, it is replaced by DestroyedWall.
    ///     Only way how to make damage is to use pickaxe.
    /// </summary>
    public class DamagedWall : DynamicTile, IInteractable
    {
        public float Health { get; private set; }

        private DamagedWall(ITilesetTable tilesetTable, Vector2I position)
            : base(tilesetTable, position)
        {
            Health = 1f;
        }

        public DamagedWall(float damage, ITilesetTable tilesetTable, Vector2I position)
            : this(tilesetTable, position)
        {
            Health -= damage;
        }

        public void ApplyGameAction(IAtlas atlas, GameAction gameAction, Vector2 position, ITilesetTable tilesetTable = null)
        {
            UsePickaxe action = gameAction as UsePickaxe;
            if (action != null)
            {
                UsePickaxe usePickaxe = action;
                Health -= usePickaxe.Damage;
            }

            if (Health <= 0f)
                atlas.ReplaceWith(new GameActorPosition(this, position), new DestroyedWall(tilesetTable));
        }
    }

    /// <summary>
    /// </summary>
    public class DestroyedWall : StaticTile
    {
        public DestroyedWall(ITilesetTable tilesetTable)
            : base(tilesetTable)
        {
        }
    }
}