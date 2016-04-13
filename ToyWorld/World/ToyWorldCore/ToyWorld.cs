using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using TmxMapSerializer.Elements;
using VRageMath;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;
using World.WorldInterfaces;

namespace World.ToyWorldCore
{
    public class ToyWorld : IWorld
    {
        private BasicAvatarMover m_basicAvatarMover;

        public Vector2I Size { get; private set; }


        public AutoupdateRegister AutoupdateRegister { get; protected set; }
        public Atlas Atlas { get; protected set; }
        public TilesetTable TilesetTable { get; protected set; }
        public IPhysics Physics { get; protected set; }


        public ToyWorld(Map tmxDeserializedMap, StreamReader tileTable)
        {
            Contract.Requires<ArgumentNullException>(tileTable != null, "Tile table cannot be null");
            Size = new Vector2I(tmxDeserializedMap.Width, tmxDeserializedMap.Height);

            AutoupdateRegister = new AutoupdateRegister();

            TilesetTable = new TilesetTable(tmxDeserializedMap, tileTable);
            Atlas = MapLoader.LoadMap(tmxDeserializedMap, TilesetTable);

            Physics = new Physics.Physics();
        }

        private void UpdatePhysics()
        {

        }

        private void UpdateCharacters()
        {
        }

        private void UpdateAvatars()
        {
            List<IAvatar> avatars = Atlas.GetAvatars();
            Physics.TransofrmControlsToMotion(avatars);
            List<IForwardMovablePhysicalEntity> forwardMovablePhysicalEntities = avatars.Select(x => x.PhysicalEntity).ToList();
            Physics.MoveMovableDirectable(forwardMovablePhysicalEntities);
        }

        public void UpdateScheduled()
        {
            foreach (IAutoupdateable actor in AutoupdateRegister.CurrentUpdateRequests)
                actor.Update(this);
            AutoupdateRegister.CurrentUpdateRequests.Clear();
            AutoupdateRegister.Tick();
        }

        public void Update()
        {
            UpdateTiles();
            UpdateAvatars();
            UpdateCharacters();
            AutoupdateRegister.UpdateItems(this);
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

        [ContractInvariantMethod]
        private void Invariants()
        {
            Contract.Invariant(Atlas != null, "Atlas cannot be null");
            Contract.Invariant(AutoupdateRegister != null, "Autoupdate register cannot be null");
        }
    }
}