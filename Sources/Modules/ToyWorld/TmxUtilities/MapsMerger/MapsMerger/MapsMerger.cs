using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TmxMapSerializer.Elements;

namespace MapsMerger
{
    internal class MapsMerger
    {
        private int objectId = 1;

        public void Merge(List<List<Map>> mergeList)
        {
            foreach (var row in mergeList)
            {
                while (row.Count > 1)
                {
                    HorizontalMerge(row[0], row[1]);
                    row.RemoveAt(1);
                }
            }

            while (mergeList.Count > 1)
            {
                VerticalMerge(mergeList[0][0], mergeList[1][0]);
                mergeList.RemoveAt(1);
            }
        }

        private void HorizontalMerge(Map map0, Map map1)
        {
            int originalWidth = map0.Width * map0.Tilewidth;
            map0.Width += map1.Width;
            foreach (var layer in map0.Layers)
            {
                var l2 = map1.Layers.Last(x => x.Name == layer.Name);

                layer.Width += l2.Width;

                var lines0 = layer.Data.RawData.Split('\n');
                var lines1 = l2.Data.RawData.Split('\n');

                var sb = new StringBuilder();
                for (var i = 0; i < lines0.Length - 2; i++)
                {
                    sb.Append(lines0[i]).Append(lines1[i]).Append("\n");
                }
                sb.Append(lines0[lines0.Length - 2]).Append(",").Append(lines1[lines0.Length - 2]).Append("\n");
                layer.Data.RawData = sb.ToString();
            }

            for (int i = 0; i < map0.ObjectGroups.Count; i++)
            {
                foreach (var tmxMapObject in map0.ObjectGroups[i].TmxMapObjects)
                {
                    tmxMapObject.Id = objectId++;
                }
                foreach (var tmxMapObject in map1.ObjectGroups[i].TmxMapObjects)
                {
                    tmxMapObject.Id = objectId++;
                    tmxMapObject.X += originalWidth;
                }
                map0.ObjectGroups[i].TmxMapObjects.AddRange(map1.ObjectGroups[i].TmxMapObjects);
            }
        }

        private void VerticalMerge(Map map0, Map map1)
        {
            int originalHeight = map0.Height * map0.Tileheight;
            map0.Height += map1.Height;
            foreach (var layer in map0.Layers)
            {
                var l2 = map1.Layers.Last(x => x.Name == layer.Name);
                layer.Height += l2.Height;
                var orig = layer.Data.RawData;
                var concated = l2.Data.RawData;
                layer.Data.RawData = orig.Remove(orig.Length - 1) + "," + concated;
            }

            for (int i = 0; i < map0.ObjectGroups.Count; i++)
            {
                foreach (var tmxMapObject in map0.ObjectGroups[i].TmxMapObjects)
                {
                    tmxMapObject.Id = objectId++;
                }
                foreach (var tmxMapObject in map1.ObjectGroups[i].TmxMapObjects)
                {
                    tmxMapObject.Id = objectId++;
                    tmxMapObject.Y += originalHeight;
                }
                map0.ObjectGroups[i].TmxMapObjects.AddRange(map1.ObjectGroups[i].TmxMapObjects);
            }
        }
    }
}
