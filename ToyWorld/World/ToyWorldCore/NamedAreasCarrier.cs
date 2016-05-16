using System;
using System.Collections.Generic;
using GoodAI.Logging;
using Utils.VRageRIP.Lib.Extensions;
using VRageMath;
using World.GameActors.Tiles;
using World.GameActors.Tiles.Path;

namespace World.ToyWorldCore
{
    public interface INamedAreasCarrier
    {
        string AreaName(Vector2 coordinates);
        string RoomName(Vector2 coordinates);
    }

    public class NamedAreasCarrier : INamedAreasCarrier
    {
        private string[][] Rooms { get; set; }
        private string[][] Areas { get; set; }

        public NamedAreasCarrier(ITileLayer pathLayer, List<Tuple<Vector2I, string>> positionsNamesTuples)
        {
            Rooms = ArrayCreator.CreateJaggedArray<string[][]>(pathLayer.Width, pathLayer.Height);
            Areas = ArrayCreator.CreateJaggedArray<string[][]>(pathLayer.Width, pathLayer.Height);

            foreach (Tuple<Vector2I, string> positionNameTuple in positionsNamesTuples)
            {
                Vector2I position = positionNameTuple.Item1;
                string name = positionNameTuple.Item2;
                FillNamedArea(position.X, position.Y, name, pathLayer);
            }
        }

        public string AreaName(Vector2 coordinates)
        {
            Vector2I v = new Vector2I(Vector2.Floor(coordinates));
            return Areas[v.X][v.Y];
        }

        public string RoomName(Vector2 coordinates)
        {
            Vector2I v = new Vector2I(Vector2.Floor(coordinates));
            return Rooms[v.X][v.Y];
        }

        private void FillNamedArea(int x, int y, string name, ITileLayer pathLayer)
        {
            Tile tile = pathLayer.GetActorAt(x, y);
            if (tile is Room)
            {
                FillRoom(x, y, name, pathLayer);
            }
            else
            {
                FillArea(x, y, name, pathLayer);
            }
        }

        private void FillArea(int x, int y, string name, ITileLayer pathLayer)
        {
            if (Areas[x][y] != null)
            {
                if (Areas[x][y] == name)
                {
                    Log.Instance.Info("Area \"" + name + "\" has two AreaLabels.");
                }
                else
                {
                    Log.Instance.Warn("Two AreaLabels in one Area: \"" + Areas[x][y] + "\", \"" + name + "\".");
                }
            }
            ExpandArea(x, y, name, pathLayer);
        }

        private void ExpandArea(int x, int y, string name, ITileLayer pathLayer)
        {
            if (Areas[x][y] != null) return;
            Tile tile = pathLayer.GetActorAt(x,y);
            if (tile is AreaBorder || tile is AreaBorderPath) return;
            Areas[x][y] = name;
            ExpandArea(x + 1, y, name, pathLayer);
            ExpandArea(x, y + 1, name, pathLayer);
            ExpandArea(x, y - 1, name, pathLayer);
            ExpandArea(x - 1, y, name, pathLayer);
        }

        private void FillRoom(int x, int y, string name, ITileLayer pathLayer)
        {
            if (Rooms[x][y] != null)
            {
                if (Rooms[x][y] == name)
                {
                    Log.Instance.Info("Room " + name + " has two AreaLabels.");
                }
                else {
                    Log.Instance.Warn("Two AreaLabels in one Room: \"" + Rooms[x][y] + "\", \"" + name + "\".");
                }
            }
            ExpandRoom(x, y, name, pathLayer);
        }

        private void ExpandRoom(int x, int y, string name, ITileLayer pathLayer)
        {
            if (Rooms[x][y] != null) return;
            if (!(pathLayer.GetActorAt(x, y) is Room)) return;
            Rooms[x][y] = name;
            ExpandRoom(x + 1, y, name, pathLayer);
            ExpandRoom(x, y + 1, name, pathLayer);
            ExpandRoom(x, y - 1, name, pathLayer);
            ExpandRoom(x - 1, y, name, pathLayer);
        }
    }
}