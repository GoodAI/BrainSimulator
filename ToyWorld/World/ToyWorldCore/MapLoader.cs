using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public static class MapLoader
    {
        /// <summary>
        /// Loads map from specified path, creates tiles and objects and put them into new Atlas object.
        /// </summary>
        /// <param name="tmxFile"></param>
        /// <param name="tilesetTable"></param>
        /// <returns>Atlas with initial state of ToyWorld</returns>
        public static Atlas LoadMap(Map map, TilesetTable tilesetTable)
        {
            TmxSerializer tmxMapSerializer = new TmxSerializer();

            Atlas atlas = new Atlas();

            foreach (LayerType layerType in Enum.GetValues(typeof(LayerType)).Cast<LayerType>())
            {
                string layerName = Enum.GetName(typeof (LayerType), layerType);

                Debug.Assert(layerName != null);

                if (layerName.Contains("Object"))
                {
                    var objectLayer = map.ObjectGroups.First(x => x.Name == layerName);
                    atlas.ObjectLayers.Add(
                        FillObjectLayer(objectLayer, layerType)
                        );
                }
                else
                {
                    var tileLayer = map.Layers.First(x => x.Name == layerName);
                    if (tileLayer == null)
                        throw new Exception("Layer " + layerName + " not found in given tmx file!");
                    atlas.TileLayers.Add(
                        FillTileLayer(tileLayer, layerType, atlas.StaticTilesContainer, tilesetTable)
                        );
                }
            }

            return atlas;
        }

        private static IObjectLayer FillObjectLayer(ObjectGroup objectLayer, LayerType layerType)
        {
//            TODO : write loading of objects
            return new SimpleObjectLayer(layerType);
        }

        private static ITileLayer FillTileLayer(Layer layer, LayerType layerType, Dictionary<int,StaticTile> staticTilesContainer, TilesetTable tilesetTable)
        {
            SimpleTileLayer newSimpleLayer = new SimpleTileLayer(layerType, layer.Width + 1, layer.Height + 1);
            var lines = layer.Data.RawData.Split('\n');
            var assembly = Assembly.GetExecutingAssembly();
            var cachedTypes = assembly.GetTypes();
            for (int i = 0; i < lines.Length; i++)
            {
                var tiles = lines[i].Split(',');
                for (int j = 0; j < tiles.Length; j++)
                {
                    if (tiles[j].Trim() == "")
                        continue;
                    var tileNumber = int.Parse(tiles[j]);
                    if (staticTilesContainer.ContainsKey(tileNumber))
                    {
                        newSimpleLayer.Tiles[i,j] = staticTilesContainer[tileNumber];
                    }
                    else
                    {
                        var tileName = tilesetTable.TileName(tileNumber);
                        if (tileName != null)
                        {
                            var newTile = CreateInstance(tileName, tileNumber, cachedTypes);
                            newSimpleLayer.Tiles[i,j] = newTile;
                            if (newTile is StaticTile)
                            {
                                staticTilesContainer.Add(tileNumber, newTile as StaticTile);
                            }
                        }
//                        TODO : before release check code below is active
//                        else
//                            Debug.Assert(false, "Tile with number " + tileNumber + " was not found in TilesetTable");
                    }

                }
            }
            return newSimpleLayer;
        }

        private static Tile CreateInstance(string className, int tileNumber, Type[] types)
        {
            for (int i = 0; i < types.Length; i++)
            {
                if (types[i].Name == className)
                    return (Tile)Activator.CreateInstance(types[i], tileNumber);
            }
            return null;
        }
    }
}
