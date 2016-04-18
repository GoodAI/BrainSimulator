using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using VRageMath;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;

namespace World.ToyWorldCore
{
    public static class MapLoader
    {
        /// <summary>
        /// Loads map from specified path, creates tiles and objects and put them into new Atlas object.
        /// </summary>
        /// <param name="map"></param>
        /// <param name="tilesetTable"></param>
        /// <returns>Atlas with initial state of ToyWorld</returns>
        public static Atlas LoadMap(Map map, TilesetTable tilesetTable, Action<GameActor> initializer)
        {
            Atlas atlas = new Atlas();

            map.Tilesets.OrderBy(x => x.Firstgid);

            foreach (LayerType layerType in Enum.GetValues(typeof(LayerType)).Cast<LayerType>())
            {
                string layerName = Enum.GetName(typeof(LayerType), layerType);

                Debug.Assert(layerName != null);

                if (layerName.Contains("Object"))
                {
                    var objectLayer = map.ObjectGroups.First(x => x.Name == layerName);
                    atlas.ObjectLayers.Add(
                        FillObjectLayer(atlas, objectLayer, layerType, initializer, map.Tilesets, map.Tilewidth, map.Tileheight));
                }
                else
                {
                    var tileLayer = map.Layers.First(x => x.Name == layerName);
                    if (tileLayer == null)
                        throw new Exception("Layer " + layerName + " not found in given .tmx file!");
                    atlas.TileLayers.Add(
                        FillTileLayer(tileLayer, layerType, atlas.StaticTilesContainer, tilesetTable, initializer)
                        );
                }
            }

            return atlas;
        }

        private static IObjectLayer FillObjectLayer(
            Atlas atlas,
            ObjectGroup objectLayer,
            LayerType layerType,
            Action<GameActor> initializer,
            List<Tileset> tilesets,
            int tileWidth,
            int tileHeight
            )
        {
            NormalizeObjectPositions(objectLayer.TmxMapObjects, tileWidth, tileHeight);

            //            TODO : write loading of objects
            var simpleObjectLayer = new SimpleObjectLayer(layerType);

            // avatars list
            IEnumerable<TmxObject> avatars = objectLayer.TmxMapObjects.Where(x => x.Type == "Avatar");

            foreach (var avatar in avatars)
            {
                var initialPosition = new Vector2(avatar.X, avatar.Y);
                var size = new Vector2(avatar.Width, avatar.Height);
                float rotation = avatar.Rotation;

                int originalGid = avatar.Gid;
                Tuple<string, int> nameNewGid = TilesetNameFromGid(tilesets, originalGid);
                string tilesetName = nameNewGid.Item1;
                int newGid = nameNewGid.Item2;

                var gameAvatar = new Avatar(tilesetName, newGid, avatar.Name, avatar.Id, initialPosition, size, rotation);

                // this is magic
                if (avatar.Properties != null)
                {
                    SetGameObjectProperties(avatar.Properties.PropertiesList, gameAvatar);
                }
                
                initializer.Invoke(gameAvatar);
                simpleObjectLayer.AddGameObject(gameAvatar);
                atlas.AddAvatar(gameAvatar);
            }

            return simpleObjectLayer;
        }

        /// <summary>
        /// From absolute position in pixels to side of tile = 1
        /// </summary>
        /// <param name="tmxObjects"></param>
        /// <param name="tileWidth"></param>
        /// <param name="tileHeight"></param>
        private static void NormalizeObjectPositions(List<TmxObject> tmxObjects, int tileWidth, int tileHeight)
        {
            foreach (var tmxObject in tmxObjects)
            {
                tmxObject.X /= tileWidth;
                tmxObject.Y /= tileHeight;
            }
        }

        private static ITileLayer FillTileLayer(
            Layer layer,
            LayerType layerType,
            Dictionary<int, StaticTile> staticTilesContainer,
            TilesetTable tilesetTable,
            Action<GameActor> initializer)
        {
            SimpleTileLayer newSimpleLayer = new SimpleTileLayer(layerType, layer.Width, layer.Height);
            var lines = layer.Data.RawData.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
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
                        newSimpleLayer.Tiles[i, j] = staticTilesContainer[tileNumber];
                    }
                    else
                    {
                        var tileName = tilesetTable.TileName(tileNumber);
                        if (tileName != null)
                        {
                            var newTile = CreateInstance(tileName, tileNumber, cachedTypes);
                            initializer.Invoke(newTile);
                            newSimpleLayer.Tiles[i, j] = newTile;
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
            // TODO : make sure next line is active before release
            //            throw new Exception("MapLoader cannot find class " + className);
            return null;
        }

        private static Tuple<string, int> TilesetNameFromGid(List<Tileset> tilesets, int gid)
        {
            if (gid < tilesets.First().Firstgid)
            {
                throw new ArgumentException(".tmx file corrupted. GID " + gid + " belongs to no tileset!");
            }

            foreach (Tileset tileset in tilesets)
            {
                if (tileset.Firstgid < gid)
                {
                    return new Tuple<string, int>(tileset.Name, gid - tileset.Firstgid + 1);
                }
            }
            throw new ArgumentException(".tmx file corrupted. GID " + gid + " belongs to no tileset!");
        }

        /// <summary>
        /// Set GameObject properties. Properties are paired with name.
        /// </summary>
        /// <param name="properties"></param>
        /// <param name="gameObject"></param>
        private static void SetGameObjectProperties(List<Property> properties, GameObject gameObject)
        {
            Type type = gameObject.GetType();
            Convert.ChangeType(gameObject, type);

            foreach (Property property in properties)
            {
                try
                {
                    PropertyInfo gameObjectProperty = type.GetProperty(property.Name);
                    Type propertyType = gameObjectProperty.PropertyType;
                    object value;
                    try
                    {
                        if (propertyType == typeof(int))
                        {
                            value = int.Parse(property.Value);
                        }
                        else if (propertyType == typeof(float))
                        {
                            value = float.Parse(property.Value, CultureInfo.InvariantCulture);
                        }
                        else if (propertyType == typeof(string))
                        {
                            value = property.Value;
                        }
                        else
                        {
                            throw new NotSupportedException("Property requires type which is not int, float or string.");
                        }
                        gameObjectProperty.SetValue(gameObject, value);
                    }
                    catch (FormatException)
                    {
                        throw new FormatException(
                            "Cannot parse property " + property.Name +
                            " with value " + property.Value +
                            " at object " + gameObject.Name +
                            " at " + gameObject.Position + ".\n" +
                            "Property type should be " + propertyType.Name);
                    }
                }
                catch (AmbiguousMatchException)
                {
                    IEnumerable<string> propertiesNames = type.GetProperties().Select(x => x.Name);
                    string joined = String.Join(",\n", propertiesNames);
                    throw new NotSupportedException(
                        ".tmx file contains unknown property " + property.Name +
                        " at object " + gameObject.Name +
                        " at " + gameObject.Position + ".\n" +
                        "Available properties are: \n" + joined);
                }
            }
        }
    }
}
