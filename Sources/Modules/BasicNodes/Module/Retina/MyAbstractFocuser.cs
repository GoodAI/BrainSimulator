using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using System.ComponentModel;
using System;
using YAXLib;
using GoodAI.Modules.VSA;

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

        public int RetinaCircles;


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
            //Output.Dims = new TensorDimensions(OutputWidth, OutputWidth);

            NumberPupilSamples = 1;
            if (PupilControl != null && PupilControl.Count > 3) // for multi input -> set how $ pupils samples from the count
                NumberPupilSamples = PupilControl.Count / PupilControl.ColumnHint;

            //Output.Dims[1] = Output.Dims.ElementCount * NumberPupilSamples / Output.Dims[0];
            Output.Dims = new TensorDimensions(OutputWidth, OutputWidth * NumberPupilSamples);

            TempPupilControl.Count = 3;

            RetinaPtsDefsMask.Count = OutputSize * 2;
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


        [MyTaskInfo(Disabled = true, OneShot = true), Description("InitRetina")]
        abstract public class MyAbstractInitRetinaTask : MyTask<MyAbstractFocuser>
        {
            public enum MyRetinaInitMode
            {
                Circles,
                GaussSampling
            }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = MyRetinaInitMode.Circles)]
            public MyRetinaInitMode RetinaInitMode { get; set; }

            public override void Execute()
            {
                switch (RetinaInitMode)
                {
                    case MyRetinaInitMode.Circles:
                        InitRetinaCirclesMasks();
                        break;
                    case MyRetinaInitMode.GaussSampling:
                        InitRetinaGaussSamplingMasks();
                        break;
                    default:
                        break;
                }
                
            }


              // Init data for retina. It needs to be ran from the observer and other class too if node is only observing and not calculating...
            public void InitRetinaGaussSamplingMasks()
            {                  
                float std = 0.12f; // standart deviation

                MyRandomPool.GenerateRandomNormalVectors(Owner.RetinaPtsDefsMask.Host, new Random(123454321), Owner.RetinaPtsDefsMask.Count, 1, 0, std * std, false);  // seed has to be constat for everything & do not normalize.
                Owner.RetinaPtsDefsMask.SafeCopyToDevice();

            }

            // Init data for retina. It needs to be ran from the observer and other class too if node is only observing and not calculating...
            public void InitRetinaCirclesMasks()
            {
                float ptsInCircle = (Owner.RetinaPtsDefsMask.Count / 2) / Owner.RetinaCircles;
                float alpha_step = 2.0f * (float)Math.PI / ptsInCircle;

                int i = 0;
                Owner.RetinaPtsDefsMask.Host[i++] = 0f; // center of the circle ;-)
                Owner.RetinaPtsDefsMask.Host[i++] = 0f;
                float radius = 1.5f; // one pixel radius
                float alpha_current = 0f;
                float alpha_start = 0f;
                while (true)
                {
                    float chord = radius * 2 * (float)Math.Sin((alpha_step / 2));  // https://en.wikipedia.org/wiki/Chord_(geometry)
                    radius += (float)Math.Pow(chord, 0.8f);  // move radius to the next circle
                    alpha_start = (alpha_start == 0) ? alpha_step / 2 : 0; // always switch between zero and step/2
                    alpha_current = alpha_start;
                    while (true)
                    {
                        Owner.RetinaPtsDefsMask.Host[i++] = radius * (float)Math.Cos(alpha_current);
                        Owner.RetinaPtsDefsMask.Host[i++] = radius * (float)Math.Sin(alpha_current);
                        alpha_current += alpha_step;
                        if (alpha_current >= 2 * Math.PI || i >= Owner.RetinaPtsDefsMask.Count)
                            break;
                    }
                    if (i >= Owner.RetinaPtsDefsMask.Count)
                        break;
                }

                // Normalize sizes to the scale of one pixel
                float norm_radius = 1 / radius * 0.9f; // norlamize to one pixel (so we can scale it then properly)
                for (i = 0; i < Owner.RetinaPtsDefsMask.Count; i++)
                {
                    Owner.RetinaPtsDefsMask.Host[i] *= norm_radius;
                }
                Owner.RetinaPtsDefsMask.SafeCopyToDevice();
            }
        }


    }
}
