using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System.ComponentModel;
using System;
using YAXLib;

namespace GoodAI.Modules.Retina
{
    /// <author>GoodAI</author>
    /// <meta>df/jk</meta>
    ///<status>Working</status>
    ///<summary>Abstract class for crops and resizes input image according to pupil control input.
    ///Pupil control input must contain position and size of focused area.</summary>
    ///<description>
    /// </description>
    public abstract class MyAbstractFocuser : MyTransform
    {
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 64)]
        public int OutputWidth { get; set; }


        [MyBrowsable, Category("RetinaTransform")]
        [YAXSerializableField(DefaultValue = 4)]
        public int RetinaCircles { get; set; }


        [MyInputBlock(1)]
        public MyMemoryBlock<float> PupilControl 
        {
            get { return GetInput(1); }
        }

        public MyMemoryBlock<float> TempPupilControl { get; set; }
        public MyMemoryBlock<float> RetinaPtsDefsMask { get; set; }
        public MyMemoryBlock<float> RetinaTempCumulateSize { get; set; }

        public int NumberPupilSamples;

        public override void UpdateMemoryBlocks()
        {
            OutputSize = OutputWidth * OutputWidth;
            Output.ColumnHint = OutputWidth;

            NumberPupilSamples = 1;
            if (PupilControl != null && PupilControl.Count > 3) // for multi input -> set how $ pupils samples from the count
                NumberPupilSamples = PupilControl.Count / PupilControl.ColumnHint;
            OutputSize *= NumberPupilSamples;

            TempPupilControl.Count = 3;

            RetinaPtsDefsMask.Count = OutputSize* 2;
            RetinaPtsDefsMask.ColumnHint = 2;
            RetinaTempCumulateSize.Count = Output.Count;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (PupilControl != null)
            {
                validator.AssertError(PupilControl.Count > 2, this, "Not enough control values (at least 3 values needed)");

                validator.AssertError((PupilControl.Count % 3) == 0, this, "Wrong pupil control input size, it has to be [x,y,s] or [x,y,s;x,y,s...]");
                validator.AssertError((float)PupilControl.Count / (float)PupilControl.ColumnHint != 0, this, "If input is matrix, it has to be 3 columns and N rows, each row x,y,s");
            }
        }

        


        // Init data for retina. It needs to be ran from the observer and other class too if node is only observing and not calculating...
        public void InitRetinaMasks()
        {
            float ptsInCircle = (RetinaPtsDefsMask.Count/2) / RetinaCircles;
            float alpha_step = 2.0f * (float)Math.PI / ptsInCircle;

            int i = 0;
            RetinaPtsDefsMask.Host[i++] = 0f; // center of the circle ;-)
            RetinaPtsDefsMask.Host[i++] = 0f;
            float radius = 1f; // one pixel radius
            float alpha_current = 0f;
            float alpha_start = 0f;
            while (true)
            {
                float chord = radius * 2 * (float)Math.Sin((alpha_step / 2));  // https://en.wikipedia.org/wiki/Chord_(geometry)
                radius += chord * 0.8f;  // move radius to the next circle
                alpha_start = (alpha_start == 0) ? alpha_step / 2 : 0; // always switch between zero and step/2
                alpha_current = alpha_start;
                while (true)
                {
                    RetinaPtsDefsMask.Host[i++] = radius * (float)Math.Cos(alpha_current);
                    RetinaPtsDefsMask.Host[i++] = radius * (float)Math.Sin(alpha_current);
                    alpha_current += alpha_step;
                    if (alpha_current >= 2 * Math.PI || i >= RetinaPtsDefsMask.Count)
                        break;
                }
                if (i >= RetinaPtsDefsMask.Count)
                    break;
            }
            RetinaPtsDefsMask.SafeCopyToDevice();
        }


    }
}
