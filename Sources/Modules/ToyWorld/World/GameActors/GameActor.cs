using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using VRageMath;
using World.Atlas.Layers;

namespace World.GameActors
{
    public interface IGameActor
    {
        int TilesetId { get; set; }
    }

    /// <summary>
    /// Common ancestor of GameObjects and Tiles
    /// </summary>
    public abstract class GameActor : IGameActor
    {
        /// <summary>
        /// Serial number of texture in tileset.
        /// </summary>
        public int TilesetId { get; set; }

        // separate copy for each final subclass

        private static readonly Dictionary<string, int> m_defaultTilesetIdDict = new Dictionary<string, int>();
        public int DefaultTextureId
        {
            get
            {
                return m_defaultTilesetIdDict.ContainsKey(GetType().Name)
                    ? m_defaultTilesetIdDict[GetType().Name]
                    : default(int);
            }
            set { m_defaultTilesetIdDict[GetType().Name] = value; }
        }

        private static readonly Dictionary<string, string> m_defaultTilesetIdNameDict = new Dictionary<string, string>();
        public string DefaultTextureName
        {
            get
            {
                return m_defaultTilesetIdNameDict.ContainsKey(GetType().Name)
                    ? m_defaultTilesetIdNameDict[GetType().Name]
                    : default(string);
            }
            set { m_defaultTilesetIdNameDict[GetType().Name] = value; }
        }

        private static readonly Dictionary<string, TilesetIds> m_tilesetIdsDict = new Dictionary<string, TilesetIds>();
        public TilesetIds AlternativeTextures
        {
            get
            {
                return m_tilesetIdsDict.ContainsKey(GetType().Name)
                    ? m_tilesetIdsDict[GetType().Name]
                    : default(TilesetIds);
            }
            set { m_tilesetIdsDict[GetType().Name] = value; }
        }

        public static Type GetType(string type)
        {
            Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
            Assembly assembly = assemblies.First(a => a.FullName.StartsWith("World,"));
            Type t = assembly.GetTypes().Where(a => a.IsSubclassOf(typeof(GameActor))).FirstOrDefault(a => a.Name == type);
            return t;
        }
    }

    public class GameActorPosition
    {
        public GameActor Actor { get; private set; }
        public Vector2 Position { get; private set; }
        public LayerType Layer { get; set; }

        public GameActorPosition(GameActor actor, Vector2 position, LayerType layer)
        {
            Actor = actor;
            Position = position;
            Layer = layer;
        }
    }

    public class TilesetIds
    {
        private List<Tuple<int, string>> IdName = new List<Tuple<int, string>>();

        public string Name(int id)
        {
            return IdName.First(x => x.Item1 == id).Item2;
        }

        public int Id(string name)
        {
            return IdName.First(x => x.Item2 == name).Item1;
        }

        public void Add(int id, string name)
        {
            IdName.Add(new Tuple<int, string>(id, name));
        }
    }
}