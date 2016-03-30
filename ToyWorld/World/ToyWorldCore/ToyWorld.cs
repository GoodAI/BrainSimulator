using System;
using World.Tiles;

namespace World.ToyWorldCore
{
    public class ToyWorld
    {

        private IPhysics Physics { get; set; }

        public World.Atlas Atlas
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }

        public World.Tiles.TileSetTableParser TileSetTableParser
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }

        public World.MapLoader MapLoader
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }

        public World.AutoupdateRegister AutoupdateRegister
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }

        private void Autoupdate()
        {
            throw new NotImplementedException();
        }

        private void MoveObjects()
        {
            throw new NotImplementedException();
        }
    }
}