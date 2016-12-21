using System;
using System.Collections.Generic;
using GoodAI.Logging;
using Utils.VRageRIP.Lib.Extensions;
using VRageMath;
using World.Atlas.Layers;
using World.GameActors.Tiles;
using World.GameActors.Tiles.Area;
using World.GameActors.Tiles.Background;
using World.ToyWorldCore;

namespace World.Atlas
{
    public interface IAreasCarrier
    {
        string AreaName(Vector2 coordinates);
        string RoomName(Vector2 coordinates);
        Room Room(Vector2 coordinates);
    }

    public class AreasCarrier : IAreasCarrier
    {
        private Room[][] Rooms { get; set; }
        private string[][] AreaNames { get; set; }

        public AreasCarrier(ITileLayer pathLayer, ITileLayer areaLayer, List<Tuple<Vector2I, string>> positionsNamesTuples)
        {
            Rooms = ArrayCreator.CreateJaggedArray<Room[][]>(pathLayer.Width, pathLayer.Height);
            AreaNames = ArrayCreator.CreateJaggedArray<string[][]>(pathLayer.Width, pathLayer.Height);

            foreach (Tuple<Vector2I, string> positionNameTuple in positionsNamesTuples)
            {
                Vector2I position = positionNameTuple.Item1;
                string name = positionNameTuple.Item2;
                FillNamedArea(position.X, position.Y, name, pathLayer, areaLayer);
            }
        }

        public string AreaName(Vector2 coordinates)
        {
            Vector2I v = new Vector2I(Vector2.Floor(coordinates));
            try
            {
                return AreaNames[v.X][v.Y];
            }
            catch (IndexOutOfRangeException)
            {
                return null;
            }

        }

        public string RoomName(Vector2 coordinates)
        {
            Room room = Room(coordinates);
            return room == null ? null : room.Name;
        }

        public Room Room(Vector2 coordinates)
        {
            Vector2I v = new Vector2I(Vector2.Floor(coordinates));
            try
            {
                return Rooms[v.X][v.Y];
            }
            catch (IndexOutOfRangeException)
            {
                return null;
            }
        }

        private void FillNamedArea(int x, int y, string name, ITileLayer pathLayer, ITileLayer areaLayer)
        {
            Tile tile = pathLayer.GetActorAt(x, y);
            if (tile is RoomTile)
            {
                FillRoom(x, y, name, pathLayer);
            }
            else
            {
                FillArea(x, y, name, areaLayer);
            }
        }

        private void FillArea(int x, int y, string name, ITileLayer areaLayer)
        {
            if (AreaNames[x][y] != null)
            {
                if (AreaNames[x][y] == name)
                {
                    Log.Instance.Info("Area \"" + name + "\" has two AreaLabels.");
                }
                else
                {
                    Log.Instance.Warn("Two AreaLabels in one Area: \"" + AreaNames[x][y] + "\", \"" + name + "\".");
                }
            }
            ExpandArea(x, y, name, areaLayer);
        }

        private void ExpandArea(int x, int y, string name, ITileLayer areaLayer)
        {
            if (AreaNames[x][y] != null) return;
            Tile tile = areaLayer.GetActorAt(x,y);
            if (tile is AreaBorder) return;
            AreaNames[x][y] = name;
            ExpandArea(x + 1, y, name, areaLayer);
            ExpandArea(x, y + 1, name, areaLayer);
            ExpandArea(x, y - 1, name, areaLayer);
            ExpandArea(x - 1, y, name, areaLayer);
        }

        private void FillRoom(int x, int y, string name, ITileLayer pathLayer)
        {
            if (Rooms[x][y] != null)
            {
                if (Rooms[x][y].Name == name)
                {
                    Log.Instance.Info("RoomTile " + name + " has two AreaLabels.");
                }
                else {
                    Log.Instance.Warn("Two AreaLabels in one RoomTile: \"" + Rooms[x][y] + "\", \"" + name + "\".");
                }
                return;
            }
            var room = new Room(name);
            ExpandRoom(x, y, room, pathLayer);
        }

        private void ExpandRoom(int x, int y, Room room, ITileLayer pathLayer)
        {
            if (Rooms[x][y] != null) return;
            if (!(pathLayer.GetActorAt(x, y) is RoomTile)) return;
            Rooms[x][y] = room;
            room.Size++;
            ExpandRoom(x + 1, y, room, pathLayer);
            ExpandRoom(x, y + 1, room, pathLayer);
            ExpandRoom(x, y - 1, room, pathLayer);
            ExpandRoom(x - 1, y, room, pathLayer);
        }
    }
}