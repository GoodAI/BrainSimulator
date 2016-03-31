using World.GameActions;

namespace World.GameActors.Tiles
{
    class OnBackground : StaticTile
    {
        public OnBackground(int tileType) : base(tileType)
        {
        }

        public OnBackground(GameAction gameAction) : base(gameAction)
        {
        }
    }
}
