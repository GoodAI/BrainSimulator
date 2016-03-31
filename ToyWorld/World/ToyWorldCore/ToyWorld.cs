using System;
using World.GameActors.Tiles;
using World.WorldInterfaces;

namespace World.ToyWorldCore
{
    public class ToyWorld : IWorld
    {
        public ToyWorld(string tmxMapFile, string tileTable = @"GameActors\Tiles\Tilesets\TilesetTable.csv")
        {
            AutoupdateRegister = new AutoupdateRegister();
            var loadedMap = MapLoader.LoadMap(tmxMapFile, new TilesetTable(tileTable));
        }

        private IPhysics Physics { get; set; }

        public TilesetTable TileSetTableParser { get; private set; }

        public AutoupdateRegister AutoupdateRegister { get; private set; }

        public Atlas Atlas { get; private set; }

        private void UpdatePhysics()
        {
            throw new NotImplementedException();
        }

        private void UpdateCharacters()
        {
            throw new System.NotImplementedException();
        }

        private void UpdateAvatars()
        {
            throw new System.NotImplementedException();
        }

        public void Update()
        {
            UpdateTiles();
            UpdateAvatars();
            UpdateCharacters();
            UpdatePhysics();
        }

        private void UpdateTiles()
        {
            throw new System.NotImplementedException();
        }
    }
}