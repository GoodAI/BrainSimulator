namespace World.GameActors.Tiles.Background
{
    public class Background : StaticTile
    {
        public Background() : base (){ } 

 		public Background(int textureId) : base(textureId) { }

        public Background(string textureName) : base(textureName)
        {
        }
    }
}
