using System;

namespace World.Tiles
{
    /// <summary>
    /// Wall can be transformed to DamagedWall if Pickaxe is used
    /// </summary>
    public class Wall : StaticTile, ITransformable
    {
        public AbstractTile TransformTo(GameAction gameAction)
        {
            if (gameAction is UsePickaxe)
                return new DamagedWall((gameAction as UsePickaxe).Damage);
            return null;
        }
    }

    /// <summary>
    /// Damaged wall has health from (0,1) excl. If health leq 0, it is replaced by DestroyedWall
    /// </summary>
    public class DamagedWall : DynamicTile, ITransformable
    {
        private float _health;

        public DamagedWall()
        {
            _health = 0.99f;
        }

        public DamagedWall(float damage) : this()
        {
            _health -= damage;
        }

        public override void Update(GameAction gameAction)
        {
            if (gameAction is UsePickaxe)
            {
                UsePickaxe usePickaxe = gameAction as UsePickaxe;
                _health -= usePickaxe.Damage;
                if (_health <= 0f)
                {
                    TransformTo(gameAction);
                }
            }
        }

        public override void RegisterForUpdate()
        {
            throw new NotImplementedException();
        }

        public AbstractTile TransformTo(GameAction gameAction)
        {
            if (gameAction is UsePickaxe)
            {
                return new DestroyedWall();
            }
            return null;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class DestroyedWall : StaticTile
    {
    }
}

