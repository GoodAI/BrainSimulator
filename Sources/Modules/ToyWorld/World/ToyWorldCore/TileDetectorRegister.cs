using System.Collections.Generic;
using System.Diagnostics;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class TileDetectorRegister
    {
        private readonly IAtlas m_atlas;

        public TileDetectorRegister(IAtlas atlas)
        {
            m_atlas = atlas;
        }

        public void Update()
        {
            IObjectLayer objectLayer = m_atlas.GetLayer(LayerType.Object) as IObjectLayer;
            Debug.Assert(objectLayer != null, "objectLayer != null");
            // for all game objects
            foreach (IGameObject gameObject in objectLayer.GetGameObjects())
            {
                List<Vector2I> coverTiles = gameObject.PhysicalEntity.CoverTiles();
                // for all covering tiles of this game objects
                foreach (Vector2I coverTile in coverTiles)
                {
                    IEnumerable<GameActorPosition> actorsAt = m_atlas.ActorsAt(new Vector2(coverTile),
                        LayerType.TileLayers);
                    foreach (GameActorPosition result in actorsAt)
                    {
                        var tileDetector = result.Actor as IDetectorTile;

                        if (tileDetector == null) continue;
                        if (tileDetector.RequiresCenterOfObject && !Atlas.Layers.Atlas.InsideTile(coverTile, gameObject.Position))
                            continue;

                        tileDetector.ObjectDetected(gameObject, m_atlas);
                    }
                }
            }
        }
    }
}