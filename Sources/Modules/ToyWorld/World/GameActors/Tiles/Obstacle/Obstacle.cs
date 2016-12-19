namespace World.GameActors.Tiles.Obstacle
{
    public class Obstacle : StaticTile
    {
        public Obstacle() : base (){ } 

 		public Obstacle(int textureId) : base(textureId) { }

        public Obstacle(string textureName)
            : base(textureName)
        {
        }
    }
}
