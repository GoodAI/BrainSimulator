using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Reflection;
using GoodAI.Logging;
using TmxMapSerializer.Elements;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
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
        /// <param name="map">Deserialized .tmx file</param>
        /// <param name="tilesetTable">Table of numbers of game object and theirs classes</param>
        /// <param name="initializer"></param>
        /// <returns>Atlas with initial state of ToyWorld</returns>
        public static IAtlas LoadMap(Map map, TilesetTable tilesetTable, Action<GameActor> initializer)
        {
            IAtlas atlas = new Atlas.Layers.Atlas();

            map.Tilesets = map.Tilesets.OrderBy(x => x.Firstgid).ToList();

            foreach (LayerType layerType in Enum.GetValues(typeof(LayerType)).Cast<LayerType>())
            {
                string layerName = Enum.GetName(typeof(LayerType), layerType);

                Debug.Assert(layerName != null);

                int type = (int)layerType;
                bool simpleType = type > 0 && (type & (type - 1)) == 0;
                if (!simpleType) continue;

                if (layerName.Contains("Object"))
                {
                    ObjectGroup objectLayer = map.ObjectGroups.FirstOrDefault(x => x.Name == layerName);
                    if (objectLayer == null)    // TMX does not contain such layer
                    {
                        Log.Instance.Warn("Layer " + layerName + " not found in given .tmx file!");
                        continue;
                    }
                    IObjectLayer filledObjectLayer = FillObjectLayer(
                        atlas,
                        objectLayer,
                        layerType,
                        initializer,
                        map.Tilesets,
                        map.Tilewidth,
                        map.Tileheight,
                        map.Height
                        );
                    atlas.ObjectLayers.Add(
                        filledObjectLayer);
                }
                else
                {
                    Layer tileLayer = map.Layers.FirstOrDefault(x => x.Name == layerName);
                    if (tileLayer == null)  // TMX does not contain such layer
                    {
                        Log.Instance.Warn("Layer " + layerName + " not found in given .tmx file!");
                        continue;
                    }
                    ITileLayer filledTileLayer = FillTileLayer(
                        tileLayer,
                        layerType,
                        atlas.StaticTilesContainer,
                        tilesetTable,
                        initializer);
                    atlas.TileLayers.Add(
                        filledTileLayer
                        );
                }
            }

            FillNamedAreas(atlas, map);

            SetTileRelations(atlas, map);

            return atlas;
        }

        private static void SetTileRelations(IAtlas atlas, Map map)
        {
            ObjectGroup foregroundObjects = map.ObjectGroups.FirstOrDefault(x => x.Name == "ForegroundObject");
            Debug.Assert(foregroundObjects != null, "foregroundObjects != null");
            List<TmxObject> tmxMapObjects = foregroundObjects.TmxMapObjects;
            IEnumerable<TmxObject> switcherToSwitchablePolylines = tmxMapObjects.Where(x => x.Type == "SwitcherToSwitchable");
            foreach (TmxObject switcherToSwitchablePolyline in switcherToSwitchablePolylines)
            {
                Polyline polyline = switcherToSwitchablePolyline.Polyline;
                if (polyline == null)
                {
                    throw new ArgumentException("Foreground object SwitcherToSwitchable is wrong type. Should be Polyline.");
                }
                List<Vector2> polylinePoints = PolylineTransform(map, switcherToSwitchablePolyline).ToList();
                Vector2 source = polylinePoints.First();
                Vector2 target = polylinePoints.Last();
                IEnumerable<GameActorPosition> sourceGameActors = atlas.ActorsAt(source);
                GameActorPosition switcherPosition = sourceGameActors.FirstOrDefault(x => x.Actor is ISwitcherGameActor);
                if (switcherPosition == null)
                {
                    Log.Instance.Error("SwitcherToSwitchable polyline expects Switcher type at [" + source.X + ";" + source.Y + "].");
                    return;
                }
                ISwitcherGameActor switcherGameActor = switcherPosition.Actor as ISwitcherGameActor;

                IEnumerable<GameActorPosition> targetGameActors = atlas.ActorsAt(target);
                GameActorPosition switchablePosition = targetGameActors.FirstOrDefault(x => x.Actor is ISwitchableGameActor);
                if (switchablePosition == null)
                {
                    Log.Instance.Error("SwitcherToSwitchable polyline expects Switchable type at [" + target.X + ";" + target.Y + "].");
                    return;
                }
                ISwitchableGameActor switchable = switchablePosition.Actor as ISwitchableGameActor;

                if (switcherGameActor != null) switcherGameActor.Switchable = switchable;
            }
        }

        private static IEnumerable<Vector2> PolylineTransform(Map map, TmxObject polyline)
        {
            float tilewidth = map.Tilewidth;
            float tileheight = map.Tileheight;
            float polX = polyline.X;
            float polY = polyline.Y;
            foreach (Vector2 point in polyline.Polyline.GetPoints())
            {
                yield return new Vector2(polX + point.X / tilewidth, polY - point.Y / tileheight);
            }
        }

        private static void FillNamedAreas(IAtlas atlas, Map map)
        {
            ObjectGroup foregroundObjects = map.ObjectGroups.First(x => x.Name == "ForegroundObject");

            List<TmxObject> tmxMapObjects = foregroundObjects.TmxMapObjects;
            List<TmxObject> areaLabels = tmxMapObjects.Where(x => x.Type.Trim() == "AreaLabel").ToList();

            IEnumerable<Vector2I> positions = areaLabels.Select(x => new Vector2I((int)Math.Floor(x.X), (int)Math.Floor(x.Y)));
            IEnumerable<string> names = areaLabels.Select(x => x.Name);
            List<Tuple<Vector2I, string>> namesPositions = positions.Zip(names, (x, y) => new Tuple<Vector2I, string>(x, y)).ToList();

            atlas.AreasCarrier = new AreasCarrier((ITileLayer)atlas.GetLayer(LayerType.Background),
                (ITileLayer)atlas.GetLayer(LayerType.Area), namesPositions);
        }

        private static IObjectLayer FillObjectLayer(
            IAtlas atlas,
            ObjectGroup objectLayer,
            LayerType layerType,
            Action<GameActor> initializer,
            List<Tileset> tilesets,
            int tileWidth,
            int tileHeight,
            int worldHeight
            )
        {
            List<TmxObject> tmxMapObjects = objectLayer.TmxMapObjects;
            tmxMapObjects.ForEach(x => NormalizeObjectPosition(x, tileWidth, tileHeight, worldHeight));
            tmxMapObjects.ForEach(TransformObjectPosition);

            var simpleObjectLayer = new SimpleObjectLayer(layerType);

            // avatars list
            List<TmxObject> avatars = tmxMapObjects.Where(x => x.Type == "Avatar").ToList();

            foreach (TmxObject avatar in avatars)
            {
                Avatar gameAvatar = LoadAgent(avatar, tilesets);
                initializer.Invoke(gameAvatar);
                simpleObjectLayer.AddGameObject(gameAvatar);
                atlas.AddAvatar(gameAvatar);
            }

            List<TmxObject> others = tmxMapObjects.Except(avatars).ToList();
            List<TmxObject> characters = others.Where(x => x.Gid != 0).ToList();

            foreach (TmxObject tmxObject in characters)
            {
                Character character = LoadCharacter(tmxObject, tilesets);
                initializer.Invoke(character);
                simpleObjectLayer.AddGameObject(character);
                atlas.Characters.Add(character);
            }
            others = others.Except(characters).ToList();

            // TODO : other objects



            return simpleObjectLayer;
        }

        private static Avatar LoadAgent(TmxObject avatar, List<Tileset> tilesets)
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

            return gameAvatar;
        }

        private static Character LoadCharacter(TmxObject tmxObject, List<Tileset> tilesets)
        {
            var initialPosition = new Vector2(tmxObject.X, tmxObject.Y);
            var size = new Vector2(tmxObject.Width, tmxObject.Height);
            float rotation = tmxObject.Rotation;

            int originalGid = tmxObject.Gid;
            Tuple<string, int> nameNewGid = TilesetNameFromGid(tilesets, originalGid);
            string tilesetName = nameNewGid.Item1;
            int newGid = nameNewGid.Item2;

            Type objectType = Type.GetType("World.GameActors.GameObjects." + tmxObject.Type);

            if (objectType == null)
            {
                throw new ArgumentException("MapLoader cannot find \"" + tmxObject.Type + "\" class.");
            }

            var gameObject = (Character)Activator.CreateInstance(
                objectType,
                tilesetName,
                newGid,
                tmxObject.Name,
                initialPosition,
                size,
                rotation
                );

            // this is magic
            if (tmxObject.Properties != null)
            {
                SetGameObjectProperties(tmxObject.Properties.PropertiesList, gameObject);
            }

            return gameObject;
        }

        /// <summary>
        /// Transforms from rotation around left lower corner to rotation around center
        /// </summary>
        /// <param name="tmxObject"></param>
        private static void TransformObjectPosition(TmxObject tmxObject)
        {
            Vector2 rotationCenter = new Vector2(tmxObject.X, tmxObject.Y);
            Vector2 size = new Vector2(tmxObject.Width / 2, tmxObject.Height / 2);
            Vector2 newPosition = RotateAroundCenter(rotationCenter + size, rotationCenter, tmxObject.Rotation);
            tmxObject.X = newPosition.X;
            tmxObject.Y = newPosition.Y;
        }

        /// <summary>
        /// From absolute position in pixels to side of tile = 1
        /// </summary>
        /// <param name="tmxObject"></param>
        /// <param name="tileWidth"></param>
        /// <param name="tileHeight"></param>
        /// <param name="worldHeight"></param>
        private static void NormalizeObjectPosition(TmxObject tmxObject, int tileWidth, int tileHeight, int worldHeight)
        {
            tmxObject.X /= tileWidth;
            tmxObject.Y /= tileHeight;
            tmxObject.Y = worldHeight - tmxObject.Y;
            tmxObject.Width /= tileWidth;
            tmxObject.Height /= tileHeight;
            tmxObject.Rotation = -MathHelper.ToRadians(tmxObject.Rotation);
        }

        private static ITileLayer FillTileLayer(
            Layer layer,
            LayerType layerType,
            Dictionary<int, StaticTile> staticTilesContainer,
            TilesetTable tilesetTable,
            Action<GameActor> initializer)
        {
            SimpleTileLayer newSimpleLayer = new SimpleTileLayer(layerType, layer.Width, layer.Height);
            string[] lines = layer.Data.RawData.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
            Assembly assembly = Assembly.GetExecutingAssembly();
            Type[] cachedTypes = assembly.GetTypes().Where(x => x.IsSubclassOf(typeof(Tile))).ToArray();
            for (int i = 0; i < lines.Length; i++)
            {
                string[] tiles = lines[i].Split(',');
                for (int j = 0; j < tiles.Length; j++)
                {
                    if (tiles[j].Trim() == "")
                        continue;
                    int tileNumber = int.Parse(tiles[j]);

                    if (tileNumber == 0) continue;

                    int x = j;
                    int y = layer.Height - 1 - i;
                    if (staticTilesContainer.ContainsKey(tileNumber))
                    {
                        newSimpleLayer.AddInternal(x, y, staticTilesContainer[tileNumber]);
                    }
                    else
                    {
                        string tileName = tilesetTable.TileName(tileNumber);
                        if (tileName != null)
                        {
                            Tile newTile = CreateInstance(tileName, tileNumber, cachedTypes, new Vector2I(x, y));
                            initializer.Invoke(newTile);
                            newSimpleLayer.AddInternal(x, y, newTile);
                            if (newTile is StaticTile)
                            {
                                staticTilesContainer.Add(tileNumber, newTile as StaticTile);
                            }
                        }
                        else
                        {
                            Log.Instance.Error("Tile with number " + tileNumber + " was not found in TilesetTable");
                        }
                    }
                }
            }

            if (layer.Properties != null)
            {
                Property render = layer.Properties.PropertiesList.FirstOrDefault(x => x.Name.Trim() == "Render");
                if (render != null)
                    newSimpleLayer.Render = bool.Parse(render.Value);
            }

            return newSimpleLayer;
        }

        private static Tile CreateInstance(string className, int tileNumber, Type[] types, Vector2I position)
        {
            foreach (Type t in types)
            {
                if (t.Name == className)
                {
                    Tile instance;

                    if (t.IsSubclassOf(typeof(DynamicTile)))
                    {
                        instance = (Tile)Activator.CreateInstance(t, tileNumber, position);
                    }
                    else
                    {
                        instance = (Tile)Activator.CreateInstance(t, tileNumber);
                    }

                    return instance;
                }
            }
            Log.Instance.Error("MapLoader cannot find class " + className);
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
        [DebuggerNonUserCode] // We want to let exceptions to be caught in this method.
        // You will receive more informative exception on call.
        private static void SetGameObjectProperties(List<Property> properties, GameObject gameObject)
        {
            Type type = gameObject.GetType();

            foreach (Property property in properties)
            {
                try
                {
                    PropertyInfo gameObjectProperty = type.GetProperty(property.Name);
                    Type propertyType = gameObjectProperty.PropertyType;
                    try
                    {
                        object value;
                        if (propertyType == typeof(int))
                        {
                            value = int.Parse(property.Value);
                        }
                        else if (propertyType == typeof(float))
                        {
                            value = float.Parse(property.Value, CultureInfo.InvariantCulture);
                        }
                        else if (propertyType == typeof(bool))
                        {
                            if (property.Value == "1")
                            {
                                value = true;
                            }
                            else if (property.Value == "0")
                            {
                                value = false;
                            }
                            else
                            {
                                value = bool.Parse(property.Value);
                            }

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
                catch (NullReferenceException)
                {
                    IEnumerable<string> propertiesNames = type.GetProperties().Select(x => x.Name);
                    string joined = String.Join(", ", propertiesNames); //String.Join(",\n", propertiesNames);
                    throw new NotSupportedException(
                        ".tmx file contains unknown property " + property.Name +
                        " at object " + gameObject.Name +
                        " at " + gameObject.Position + ".\n" +
                        "Available properties are: \n" + joined);
                }
            }
        }

        private static Vector2 RotateAroundCenter(Vector2 target, Vector2 center, float angle)
        {
            Vector2 diff = target - center;
            diff.Rotate(angle);
            return diff + center;
        }
    }
}
