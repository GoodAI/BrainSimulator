using VRageMath;
using World.Physics;

namespace World.GameActors.GameObjects
{
    /// <summary>
    /// Game object is loaded from .tmx file. According to type attribute is casted to the concrete object type.
    /// </summary>
    public interface IGameObject
    {
        /// <summary>
        /// Name of an GameObject. Deserialized from .tmx file.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Object from Physics, which stores physical properties of an GameObject.
        /// </summary>
        IPhysicalEntity PhysicalEntity { get; set; }

        /// <summary>
        /// Size is getter and setter to property inside PhysicalEntity
        /// </summary>
        Vector2 Size { get; }

        /// <summary>
        /// Position is getter and setter to property inside PhysicalEntity
        /// </summary>
        Vector2 Position { get; set; }

        /// <summary>
        /// Weight of this object.
        /// </summary>
        float Weight { get; set; }

        /// <summary>
        /// Serial number of texture in tileset.
        /// </summary>
        int TilesetId { get; set; }

        /// <summary>
        /// Name of source Tileset for texture.
        /// </summary>
        string TilesetName { get; set; }
    }

    public abstract class GameObject : GameActor, IGameObject
    {
        public string Name { get; protected set; }
        public IPhysicalEntity PhysicalEntity { get; set; }

        public Vector2 Size
        {
            get { return PhysicalEntity.CoverRectangle().Size; }
        }

        public Vector2 Position
        {
            get
            {
                return PhysicalEntity.Position;
            }
            set
            {
                PhysicalEntity.Position = value;
            }
        }

        public float Weight { get { return PhysicalEntity.Weight; } set { PhysicalEntity.Weight = value; } }

        public int TilesetId { get; set; }

        public string TilesetName { get; set; }

        public GameObject(string tilesetName, int tilesetID, string name)
        {
            TilesetId = tilesetID;
            TilesetName = tilesetName;
            Name = name;
        }
    }
}