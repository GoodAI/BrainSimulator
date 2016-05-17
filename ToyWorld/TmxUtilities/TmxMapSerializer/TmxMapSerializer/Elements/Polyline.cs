using System.Collections.Generic;
using System.Globalization;
using System.Xml.Serialization;
using VRageMath;

namespace TmxMapSerializer.Elements
{
    public class Polyline
    {
        [XmlAttribute("points")]
        public string Points { get; set; }

        public IEnumerable<Vector2> GetPoints()
        {
            string[] tuples = Points.Split(' ');
            foreach (string tuple in tuples)
            {
                string[] numbers = tuple.Split(',');
                yield return new Vector2(float.Parse(numbers[0], new CultureInfo(1)), float.Parse(numbers[1], new CultureInfo(1)));
            }
        }
    }
}