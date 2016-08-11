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

        public bool ShapesRelevant;
        public bool ColorsRelevant;
        public bool MovementRelevant;
        public bool EatRelevant;

        public AvatarsActions(bool shapesRelevant, bool colorsRelevant, bool movementRelevant, bool eatRelevant)
        {
            ShapesRelevant = shapesRelevant;
            ColorsRelevant = colorsRelevant;
            MovementRelevant = movementRelevant;
            EatRelevant = eatRelevant;
        }

        public void WriteActions(StreamWriter streamWriter)
        {
            StringBuilder stringBuilder = new StringBuilder();

            if (ShapesRelevant) stringBuilder.Append(BoolArrayToString(Shapes)).Append(",");
            else stringBuilder.Append(BoolArrayToNonSig(Shapes)).Append(",");
            if (ColorsRelevant) stringBuilder.Append(BoolArrayToString(Colors)).Append(",");
            else stringBuilder.Append(BoolArrayToNonSig(Colors)).Append(",");
            if (MovementRelevant) stringBuilder.Append(BoolArrayToString(Movement)).Append(",");
            else stringBuilder.Append(BoolArrayToNonSig(Movement)).Append(",");
            if (EatRelevant) stringBuilder.Append(Eat ? "1" : "0").AppendLine();
            else stringBuilder.Append("2").AppendLine();

            streamWriter.Write(stringBuilder);
        }

        private string BoolArrayToString(bool[] ba)
        {
            return string.Join(",", ba.Select(b => b ? "1" : "0"));
        }

        private string BoolArrayToNonSig(bool[] ba)
        {
            return string.Join(",", ba.Select(b => "2"));
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
