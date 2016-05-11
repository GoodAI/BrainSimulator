using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using GoodAI.Logging;
using TmxMapSerializer.Elements;
using VRageMath;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.Physics;
using World.WorldInterfaces;

namespace World.ToyWorldCore
{
    public class ToyWorld : IWorld
    {
        private ICollisionResolver m_collisionResolver;

        public Vector2I Size { get; private set; }
        public AutoupdateRegister AutoupdateRegister { get; protected set; }
        public TileDetectorRegister TileDetectorRegister { get; protected set; }
        public Atlas Atlas { get; protected set; }
        public TilesetTable TilesetTable { get; protected set; }
        public IPhysics Physics { get; protected set; }
        public List<Func<IAtlas, float>> SignalDispatchers { get; protected set; }

        public ToyWorld(Map tmxDeserializedMap, StreamReader tileTable)
        {
            if (tileTable == null)
                throw new ArgumentNullException("tileTable");
            if (tmxDeserializedMap == null)
                throw new ArgumentNullException("tmxDeserializedMap");
            Contract.EndContractBlock();

            Size = new Vector2I(tmxDeserializedMap.Width, tmxDeserializedMap.Height);
            AutoupdateRegister = new AutoupdateRegister();
            TilesetTable = new TilesetTable(tmxDeserializedMap, tileTable);

            InitAtlas(tmxDeserializedMap);
            InitPhysics();
            TileDetectorRegister = new TileDetectorRegister(Atlas, TilesetTable);
            RegisterSignals();
        }

        private void RegisterSignals()
        {
            Func<IAtlas, float> inventoryItem = x =>
            {
                IPickable tool = x.GetAvatars().First().Tool;
                return tool != null ? tool.TilesetId : 0;
            };

            SignalDispatchers = new List<Func<IAtlas, float>> { inventoryItem };
        }

        private void InitAtlas(Map tmxDeserializedMap)
        {
            Action<GameActor> initializer = delegate(GameActor actor)
            {
                IAutoupdateable updateable = actor as IAutoupdateable;
                if (updateable != null && updateable.NextUpdateAfter > 0)
                    AutoupdateRegister.Register(updateable, updateable.NextUpdateAfter);
            };
            Atlas = MapLoader.LoadMap(tmxDeserializedMap, TilesetTable, initializer);
        }

        private void InitPhysics()
        {
            Physics = new Physics.Physics();

            IMovementPhysics movementPhysics = new MovementPhysics();
            ICollisionChecker collisionChecker = new CollisionChecker(Atlas);

            // TODO MICHAL: setter for physics implementation
            /*
            m_collisionResolver = new NaiveCollisionResolver(collisionChecker, movementPhysics);
            /*/
            m_collisionResolver = new MomentumCollisionResolver(collisionChecker, movementPhysics);
            //*/

            Log.Instance.Debug("World.ToyWorldCore.ToyWorld: Loading Successful");
        }


        //
        // TODO: methods below will be moved to some physics class
        //


        private void UpdatePhysics()
        {
            m_collisionResolver.ResolveCollisions();
        }

        private void UpdateCharacters()
        {
            List<ICharacter> characters = Atlas.Characters;
            List<IForwardMovablePhysicalEntity> forwardMovablePhysicalEntities = characters.Select(x => x.PhysicalEntity).ToList();
            Physics.MoveMovableDirectable(forwardMovablePhysicalEntities);
        }

        private void UpdateAvatars()
        {
            List<IAvatar> avatars = Atlas.GetAvatars();
            Physics.TransformControlsPhysicalProperties(avatars);
            List<IForwardMovablePhysicalEntity> forwardMovablePhysicalEntities = avatars.Select(x => x.PhysicalEntity).ToList();
            Physics.MoveMovableDirectable(forwardMovablePhysicalEntities);
        }

        private void UpdateScheduled()
        {
            TileDetectorRegister.Update();
            AutoupdateRegister.UpdateItems(Atlas, TilesetTable);
            AutoupdateRegister.Tick();
        }

        public void Update()
        {
            UpdateScheduled();
            UpdateTiles();
            UpdateAvatars();
            UpdateCharacters();
            AutoupdateRegister.UpdateItems(Atlas, TilesetTable);
            UpdatePhysics();
            Log.Instance.Debug("World.ToyWorldCore.ToyWorld: Step performed");
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