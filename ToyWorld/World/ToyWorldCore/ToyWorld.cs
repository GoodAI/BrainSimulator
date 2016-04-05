using System;
using System.IO;
using System.Linq;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.WorldInterfaces;

namespace World.ToyWorldCore
{
    public class ToyWorld : IWorld
    {
        public ToyWorld(StreamReader tmxMapFile, StreamReader tileTable)
        {
            AutoupdateRegister = new AutoupdateRegister();
            Atlas = MapLoader.LoadMap(tmxMapFile, new TilesetTable(tileTable));
        }

        private IPhysics Physics { get; set; }

        public TilesetTable TileSetTable { get; private set; }

        public AutoupdateRegister AutoupdateRegister { get; private set; }

        public Atlas Atlas { get; private set; }

        private void UpdatePhysics()
        {
        }

        private void UpdateCharacters()
        {
        }

        private void UpdateAvatars()
        {
        }

        public void Update()
        {
            UpdateTiles();
            UpdateAvatars();
            UpdateCharacters();
            UpdatePhysics();
        }

        public int[] GetAvatarsIds()
        {
            return Atlas.Avatars.Select(avatar => avatar.Id).ToArray();
        }

        public int[] GetAvatarsNames()
        {
            return Atlas.Avatars.Select(avatar => avatar.Id).ToArray();
        }

        public Avatar GetAvatar(int id)
        {
            return null;
        }

        private void UpdateTiles()
        {
        }
    }
}