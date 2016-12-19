namespace World.GameActors.Tiles.Background
{
    public class PathTile : StaticTile
    {
        public PathTile()
        {
        }

        public PathTile(int textureId) : base(textureId) { }

        public PathTile(string textureName)
            : base(textureName)
        {
        }
    }
}