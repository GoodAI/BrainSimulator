using VRageMath;

namespace RenderingBase.RenderObjects.Textures
{
    public class TilesetImage
    {
        public TilesetImage(string imagePath, Vector2I tileSize, Vector2I tileMargin, Vector2I tileBorder)
        {
            ImagePath = imagePath;
            TileSize = tileSize;
            TileMargin = tileMargin;
            TileBorder = tileBorder;
        }

        public readonly string ImagePath; // path to the tileset image (.png)
        public Vector2I TileSize; // width and height of a tile
        public Vector2I TileMargin; // number of pixels that separate one tile from another
        public Vector2I TileBorder; // pixels that should be copied and added on each side of the tile 
        // because of correct texture scaling (linear upscaling and downscaling)
    }
}
