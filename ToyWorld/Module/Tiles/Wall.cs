using System;

namespace GoodAI.ToyWorldAPI.Tiles
{
    /// <summary>
    /// 
    /// </summary>
    public class Wall : StaticTile
    {
        private static int _tileSet;
        public override int TileType
        {
            get
            {
                if()
                return _tileSet;
            }
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class DamagedWall : DynamicTile
    {
        public override int TileType
        {
            get { throw new NotImplementedException(); }
        }

        public override void Update()
        {
            throw new NotImplementedException();
        }

        public override void RegisterForUpdate()
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public class DestroyedWall : StaticTile
    {
        public override int TileType
        {
            get { throw new NotImplementedException(); }
        }
    }
}

