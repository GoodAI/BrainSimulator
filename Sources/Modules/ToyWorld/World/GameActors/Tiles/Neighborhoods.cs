using System.Collections.Generic;
using VRageMath;

namespace World.GameActors.Tiles
{
    public static class Neighborhoods
    {
        public static IEnumerable<Vector2I> ChebyshevNeighborhood(Vector2I position, int range = 1)
        {
            for (int i = -range; i <= range; ++i)
                for (int j = -range; j <= range; ++j)
                    if (!(i == 0 && j == 0))    // omit center position
                        yield return new Vector2I(position.X + i, position.Y + j);
        }
    }
}
