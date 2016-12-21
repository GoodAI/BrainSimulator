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


        public static IEnumerable<Vector2I> VonNeumannNeighborhood(Vector2I position, int range = 1)
        {
            int x = position.X;
            int y = position.Y - range;
            List<Vector2I> enumerable = new List<Vector2I>();
            int range2 = range * 2;
            for (int i = 0; i <= range * 2; i++)
            {
                Vector2I center = new Vector2I(x, y + i);
                int count;
                if (i <= range)
                {
                    count = i * 2 + 1;
                }
                else
                {
                    count = (range2 - i) * 2 + 1;
                }
                enumerable.AddRange(CreateRowFromCenter(center, count));
            }
            return enumerable;
        }

        private static IEnumerable<Vector2I> CreateRow(Vector2I start, int count)
        {
            for (int i = start.X; i < start.X + count; i++)
            {
                yield return new Vector2I(i, start.Y);
            }
        }

        private static IEnumerable<Vector2I> CreateRowFromCenter(Vector2I center, int count)
        {
            return CreateRow(new Vector2I(center.X - count / 2, center.Y), count);
        }

    }
}
