using World.GameActions;

namespace World.GameActors.Tiles
{
    class Background : StaticTile
    {
        public Background(int tileType) : base(tileType)
        {
        }

        public Background(GameAction gameAction) : base(gameAction)
        {
        }
    }
}
