using System.Collections.Generic;
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
        private BasicAvatarMover m_basicAvatarMover;

        public ToyWorld(Map tmxDeserializedMap, StreamReader tileTable)
        {
            AutoupdateRegister = new AutoupdateRegister();

            m_tilesetTable = new TilesetTable(tileTable);
            Atlas = MapLoader.LoadMap(tmxDeserializedMap, m_tilesetTable);

            m_physics = new Physics.Physics();
        }

        public AutoupdateRegister AutoupdateRegister { get; private set; }

        public Atlas Atlas { get; private set; }

        private readonly TilesetTable m_tilesetTable;

        private readonly IPhysics m_physics;

        private void UpdatePhysics()
        {
            
        }

        private void UpdateCharacters()
        {
        }

        private void UpdateAvatars()
        {
            List<IAvatar> avatars = Atlas.GetAvatars();
            m_physics.TransofrmControlsPhysicalProperties(avatars);
            List<IForwardMovablePhysicalEntity> forwardMovablePhysicalEntities = avatars.Select(x => x.PhysicalEntity).ToList();
            m_physics.MoveMovableDirectable(forwardMovablePhysicalEntities);
        }

        public void Update()
        {
            UpdateTiles();
            UpdateAvatars();
            UpdateCharacters();
            UpdatePhysics();
        }

        public List<int> GetAvatarsIds()
        {
            return Atlas.Avatars.Keys.ToList();
        }

        public List<string> GetAvatarsNames()
        {
            return Atlas.Avatars.Values.Select(x => x.Name).ToList();
        }

        public IAvatar GetAvatar(int id)
        {
            return Atlas.Avatars[id];
        }

        private void UpdateTiles()
        {
        }
    }
}