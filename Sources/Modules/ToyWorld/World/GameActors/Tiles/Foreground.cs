namespace World.GameActors.Tiles
{
    public class Foreground : StaticTile
    {
        public Foreground() : base (){ } 

 		public Foreground(int textureId) : base(textureId) { }

        public Foreground(string textureName) : base(textureName)
        {
        }
    }
}
