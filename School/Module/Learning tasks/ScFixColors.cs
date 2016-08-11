using System;
using System.Collections.Generic;
using System.Drawing;
using ManagedCuda.NPP;

namespace GoodAI.Modules.School.LearningTasks
{
    public class ScFixColors
    {
        public List<Color> Colors;

        Color defaultColor = Color.Black;

        public ScFixColors(int num)
            : this(num, Color.Black)
        {
            
        }

        public ScFixColors(int num, Color backgroundColor)
        {
            Colors = new List<Color>();

            // convert to grayscale:
            int bkgavg = (backgroundColor.B + backgroundColor.G + backgroundColor.R) / 3;
            backgroundColor = Color.FromArgb(bkgavg, bkgavg, bkgavg);

            int minColorDiffFromBackground = 255;
            int minColorDiffFromBackgroundIndex = 0;

            for (int i = 0; i < num+1; i++)
            {
                int color = 255*i/num;
                Colors.Add(Color.FromArgb(color,color,color));
                if (Math.Abs(bkgavg - color) < minColorDiffFromBackground)
                {
                    minColorDiffFromBackground = bkgavg - color;
                    minColorDiffFromBackgroundIndex = i;
                }
            }

            Colors.RemoveAt(minColorDiffFromBackgroundIndex);
        }

        public Color GetRandomColor(Random rnd)
        {
            int index;
            return GetRandomColor(rnd, out index);
        }

        public Color GetRandomColor(Random rnd, out int index)
        {
            index = rnd.Next(Colors.Count);
            var c = Colors[index];
            return c;
        }
    }
}