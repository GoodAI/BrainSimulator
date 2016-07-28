using System;
using System.IO;
using System.Linq;
using System.Text;
using GoodAI.Modules.School.LearningTasks;

namespace GoodAI.School.Learning_tasks
{
    public class AvatarsActions
    {
        public bool[] Shapes = new bool[ScConstants.numShapes];
        public bool[] Colors = new bool[ScConstants.numColors];
        public bool[] Movement = new bool[4];
        public bool Eat;

        public void WriteActions(StreamWriter streamWriter)
        {
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.Append(BoolArrayToString(Shapes)).Append(",");
            stringBuilder.Append(BoolArrayToString(Colors)).Append(",");
            stringBuilder.Append(BoolArrayToString(Movement)).Append(",");
            stringBuilder.Append(Eat ? "1" : "0").AppendLine();

            streamWriter.Write(stringBuilder);
        }

        private string BoolArrayToString(bool[] ba)
        {
            return string.Join(",", ba.Select(b => b ? "1" : "0"));
        }

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(BoolArrayToString(Shapes)).Append(",");
            stringBuilder.Append(BoolArrayToString(Colors)).Append(",");
            stringBuilder.Append(BoolArrayToString(Movement)).Append(",");
            stringBuilder.Append(Eat ? "1" : "0");
            return stringBuilder.ToString();
        }
    }
}
