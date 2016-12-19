namespace World.GameActors.Tiles.Area
{
    public class AreaBorder : StaticTile
    {
        public AreaBorder() : base (){ } 

 		public AreaBorder(int textureId) : base(textureId) { }

        public AreaBorder(string textureName) : base(textureName)
        {
        }
    }
}