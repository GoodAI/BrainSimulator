using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public class MapLoader
    {

        /// <summary>
        /// Loads map from specified path, creates tiles and objects and put them into new Atlas object.
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="tilesetTable"></param>
        /// <returns>Atlas with initial state of ToyWorld</returns>
        public Atlas LoadMap(string fileName, TilesetTable tilesetTable)
        {
            TmxSerializer tmxMapSerializer = new TmxSerializer();

            Map map = (Map)tmxMapSerializer.Deserialize(new FileStream(fileName, FileMode.Open));

            Atlas atlas = new Atlas();

            var layersNames = Enum.GetNames(typeof (LayerType));

            var tileLayerNames = layersNames.Where(x => x.Contains("Object"));

            foreach (var tileLayer in Enum.GetValues(typeof (LayerType)))
            {
                var layerType = (LayerType) tileLayer;
                var layerName = Enum.GetName(typeof (LayerType), layerType);

                Debug.Assert(layerName != null);

                if (layerName.Contains("Object"))
                {
                    var layer = map.ObjectGroups.First(x => x.Name == layerName);
                    var newSimpleLayer = new SimpleObjectLayer(layerType);
                    atlas.ObjectLayers.Add(newSimpleLayer);
                    foreach (var tmxMapObject in layer.TmxMapObjects)
                    {
                        //tmxMapObject
                    }
                }
                else
                {
                    var layer = map.Layers.First(x => x.Name == layerName);
                    if (layer == null)
                        throw new Exception("Layer " + layerName + " not found in " + fileName + " file!");

                    var newSimpleLayer = new SimpleTileLayer(layerType, map.Width + 1, map.Height + 1);
                    atlas.TileLayers.Add(newSimpleLayer);

                    var lines = layer.Data.RawData.Split('\n');
                    for (int i = 0; i < lines.Length; i++)
                    {
                        var tiles = lines[i].Split(',');
                        for (int j = 0; j < tiles.Length; j++)
                        {
                            if (tiles[j] == "")
                                continue;
                            var tileNumber = int.Parse(tiles[j]);

                            var tileName = tilesetTable.TileName(tileNumber);
                            if (tileName != null)
                                newSimpleLayer.Tiles[i][j] = (Tile)CreateInstance(tileName, tileNumber);
//                          TODO : before release check code below is active
//                            else
//                                Debug.Assert(false, "Tile with number " + tileNumber + " was not found in TilesetTable");
                        }
                    }
                }
            }

            return atlas;
        }

        private static object CreateInstance(string className, int tileNumber)
        {
            var assembly = Assembly.GetExecutingAssembly();

            var type = assembly.GetTypes().First(t => t.Name == className);

            return Activator.CreateInstance(type, tileNumber);
        }
    }
}
