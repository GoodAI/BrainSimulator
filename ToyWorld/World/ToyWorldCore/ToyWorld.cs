using System.IO;
using System.Linq;
using TmxMapSerializer.Elements;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;
using World.WorldInterfaces;

namespace World.ToyWorldCore
{
    public class ToyWorld : IWorld
    {
        public ToyWorld(Map tmxDeserializedMap, StreamReader tileTable)
        {
            AutoupdateRegister = new AutoupdateRegister();
            Atlas = MapLoader.LoadMap(tmxDeserializedMap, new TilesetTable(tileTable));
        }

        public AutoupdateRegister AutoupdateRegister { get; private set; }

        public Atlas Atlas { get; private set; }

        public World.GameActors.Tiles.TilesetTable TilesetTable
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
            }
        }

        public IPhysics IPhysics
        {
            get
            {
                throw new System.NotImplementedException();
            }
            set
            {
                throw new System.NotImplementedException();
            }
        }

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