using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;


namespace GoodAI.Modules.Motor
{
    struct Position
    {
        public int X;
        public int Y;
    }

    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>World simulating robotic arm</summary>
    /// <description> Simulation of a robotic arm in 2D environment. <br />
    /// Parameters:
    ///              <ul>
    ///                 <li>JOINTS: Number of arm's joints, length of arm parts is automatically determined by golden ratio</li>
    ///                 <li>WORLD_WIDTH: Width of the visualised world</li>
    ///                 <li>WORLD_HEIGHT: Height of the visualised world</li>
    ///              </ul>
    /// I/O:
    ///              <ul>
    ///                 <li>MusclesInput: Torque to be applied by muscles on each joint, positive values rotate clockwise</li>
    ///                 <li>ResetInput: Optional, values other than zero reset arm to starting configuration</li>
    ///                 <li>VirtualMusclesLengthInput: Optional, visualises passed muscles lengths in VirtualOutput without affectng the "real" arm</li>
    ///                 <li>HighlightPointInput: Optional, expects 2D position of a point to be highlighted in visual output</li>
    ///                 <li>VisualOutput: Visualisation of the arm in 2D world</li>
    ///                 <li>MusclesLengthOutput: Length of muscles pulling clockwise, from minimum of 0 to maximum of 1</li>
    ///                 <li>JointsPositionOutput: 2D position of all joints</li>
    ///                 <li>EndPosition: 2D position of end effector</li>
    ///                 <li>VirtualOutput: Visualisation of virtual arm</li>
    ///              </ul>
    /// </description>
    class MyArmWorld : MyWorld
    {
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 192)]
        public int WORLD_WIDTH { get; set; }
        
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 256)]
        public int WORLD_HEIGHT { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 3)]
        public int JOINTS { get; set; }

        public float[] JointMinRotation = { 0, 320, 0, 320, 0, 0, 0, 0 };
        public float[] JointMaxRotation = { 0, 160, 160, 40, 160, 160, 160, 160 };

        public Position[] JointPosition;
        public float[] JointRotation;
        public float[] JointMomentum;
        public int[] BoneLength;

        public Position[] VirtualJointPosition;


        [MyInputBlock]
        public MyMemoryBlock<float> MusclesInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> ResetInput
        {
            get { return GetInput(1); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> VirtualMusclesLengthInput
        {
            get { return GetInput(2); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> HighlightPointInput
        {
            get { return GetInput(3); }
        }

		public MyMemoryBlock<float> Reach { get; private set; }

        [MyOutputBlock]
        public MyMemoryBlock<float> VisualOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> MusclesLengthOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> JointsPositionOutput
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> EndPosition
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> VirtualOutput
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        public MyArmWorld() {}

        public MyArmTask PerformArmStep { get; private set; }

        private void Init()
        {
            JointPosition = new Position[JOINTS + 1]; //there is an invisible joint at the end of arm. It is invisible and only used to simplify calculations
            JointRotation = new float[JOINTS + 1];
            JointMomentum = new float[JOINTS + 1];
            BoneLength = new int[JOINTS];

            VirtualJointPosition = new Position[JOINTS + 1];

            int armLength = WORLD_HEIGHT / 2 - 20;

            //divide arm into parts so that their length matches golden ratio (every bone is 1.61804 times longer than previous one)
            float totalUnitLength = 0;
            for (int i = 0; i < JOINTS; i++)
            {
                totalUnitLength += (float)Math.Pow(1.61804, i);  // == ((float)Math.Pow(1.61804, JOINTS) - 1)/0.61804;   //Guess why :)
            }
            float unitLength = armLength / totalUnitLength;

            for (int i = 0; i < JOINTS; i++)
            {
                BoneLength[i] = (int)(Math.Pow(1.61804, JOINTS - 1 - i) * unitLength);
            }

            JointPosition[0].X = 2 * WORLD_WIDTH / 3;
            JointPosition[0].Y = WORLD_HEIGHT / 2;
            JointRotation[0] = 90;
            JointMomentum[0] = 0;

            for (int i = 1; i < JOINTS + 1; i++)
            {
                JointPosition[i].X = WORLD_WIDTH - 10;
                JointPosition[i].Y = JointPosition[i - 1].Y + BoneLength[i - 1];
                JointRotation[i] = 0;
                JointMomentum[i] = 0;
            }
        }

        /// <summary>Simulates the arm<br />
        /// Parameters:
        ///              <ul>
        ///                 <li>DRAW: If set to true, end effector draws in the environment</li>
        ///                 <li>UPDATE_VISUAL_OUTPUT: If set to false, visual output is not updated</li>
        ///                 <li>GRAVITY_FORCE: Gravity force</li>
        ///                 <li>MUSCLE_FORCE: Maximum muscle force</li>
        ///                 <li>FRICTION: Friction</li>
        ///                 <li>HIGHLIGHT_X: X coordinate of a point highlighted in environment</li>
        ///                 <li>HIGHLIGHT_Y: Y coordinate of a point highlighted in environment</li>
        ///              </ul>
        /// </summary>
        [Description("Perform arm step")]
        public class MyArmTask : MyTask<MyArmWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = false)]
            public bool DRAW { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = true)]
            public bool UPDATE_VISUAL_OUTPUT { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.2f)]
            public float GRAVITY_FORCE { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.4f)]
            public float MUSCLE_FORCE { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.1f)]
            public float FRICTION { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = -1)]
            public int HIGHLIGHT_X { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = -1)]
            public int HIGHLIGHT_Y { get; set; }

            public override void Init(Int32 nGPU) {}

            public override void Execute()
            {
                if (SimulationStep == 0) Owner.Init();
                if (Owner.ResetInput != null)
                {
                    Owner.ResetInput.SafeCopyToDevice();
                    if (Owner.ResetInput.Host[0] != 0)
                    {
                        Owner.Init();
                    }
                }

                Owner.MusclesInput.SafeCopyToHost();

                float rotation = Owner.JointRotation[0]; //sum of rotations of all previous joins to calculate real rotation
                float momentumRemainder = 0.0f; //when joint reaches min/max rotation, remaining momentum is transfered to following joint

                for (int i = 1; i < Owner.JOINTS+1; i++)
                {
                    rotation += Owner.JointRotation[i];

                    float momentumChange = momentumRemainder;
                    momentumChange += GRAVITY_FORCE * (float)Math.Cos(rotation * (Math.PI / 180)); //apply gravity force
                    momentumChange -= FRICTION * Owner.JointMomentum[i]; //apply friction
                    momentumChange += MUSCLE_FORCE * Owner.MusclesInput.Host[(i - 1)]; //apply muscle force

                    Owner.JointMomentum[i] += momentumChange;
                    Owner.JointRotation[i] += Owner.JointMomentum[i];

                    if (Owner.JointMinRotation[i] <= Owner.JointMaxRotation[i]) //continuous interval
                    {
                        if (Owner.JointRotation[i] < Owner.JointMinRotation[i] || Owner.JointRotation[i] > Owner.JointMaxRotation[i]) //rotation outside allowed interval
                        {
                            momentumRemainder = Owner.JointMomentum[i] - momentumChange; //save momentum (without current change) to be applied on following joint
                            Owner.JointMomentum[i] = 0;
                            Owner.JointRotation[i] = Math.Min(Math.Max(Owner.JointRotation[i], Owner.JointMinRotation[i]), Owner.JointMaxRotation[i]);
                        }
                        else
                        {
                            momentumRemainder = 0.0f;
                        }
                        
                        //normalize angle
                        Owner.JointRotation[i] = Owner.JointRotation[i] % 360;
                        if (Owner.JointRotation[i] < 0) Owner.JointRotation[i] += 360;

                        //update muscle length
                        float oldLength = Owner.MusclesLengthOutput.Host[i - 1];
                        Owner.MusclesLengthOutput.Host[i-1] = (Owner.JointMaxRotation[i] - Owner.JointRotation[i]) / (Owner.JointMaxRotation[i] - Owner.JointMinRotation[i]);
                    }
                    else //two intervals
                    {
                        //normalize angle
                        Owner.JointRotation[i] = Owner.JointRotation[i] % 360;
                        if (Owner.JointRotation[i] < 0) Owner.JointRotation[i] += 360;

                        if (Owner.JointRotation[i] > Owner.JointMaxRotation[i] && Owner.JointRotation[i] < Owner.JointMinRotation[i]) //rotation outside allowed interval
                        {
                            momentumRemainder = Owner.JointMomentum[i] - momentumChange; //save momentum (without current change) to be applied on following joint
                            Owner.JointMomentum[i] = 0;
                            float middle = Owner.JointMaxRotation[i] + (Owner.JointMinRotation[i] - Owner.JointMaxRotation[i]) / 2;
                            //set rotation to the closest interval border
                            Owner.JointRotation[i] = Owner.JointRotation[i] > middle ? Owner.JointMinRotation[i] : Owner.JointMaxRotation[i];
                        }
                        else
                        {
                            momentumRemainder = 0.0f;
                        }

                        //update muscle length
                        if (Owner.JointRotation[i] <= Owner.JointMaxRotation[i]) Owner.JointRotation[i] += 360; //places first interval after the other
                        {
                            float oldLength = Owner.MusclesLengthOutput.Host[i - 1];
                            Owner.MusclesLengthOutput.Host[i - 1] = 1 - (Owner.JointRotation[i] - Owner.JointMinRotation[i]) / (360 - Owner.JointMinRotation[i] + Owner.JointMaxRotation[i]);
                        }
                    }

                    //update joint positions
                    Owner.JointPosition[i].X = Owner.JointPosition[i - 1].X + (int)(Math.Cos(rotation * (Math.PI / 180)) * Owner.BoneLength[i - 1]);
                    Owner.JointPosition[i].Y = Owner.JointPosition[i - 1].Y + (int)(Math.Sin(rotation * (Math.PI / 180)) * Owner.BoneLength[i - 1]);
                }

                if (DRAW)
                {
                    if (Owner.JointPosition[Owner.JOINTS].X >= 0 && Owner.JointPosition[Owner.JOINTS].X < Owner.WORLD_WIDTH && Owner.JointPosition[Owner.JOINTS].Y >= 0 && Owner.JointPosition[Owner.JOINTS].Y < Owner.WORLD_HEIGHT)
                    {
                        Owner.Reach.Host[Owner.JointPosition[Owner.JOINTS].Y * Owner.WORLD_WIDTH + Owner.JointPosition[Owner.JOINTS].X] += 0.01f;
                    }
                }

                if (UPDATE_VISUAL_OUTPUT)
                {
                    //clear whole visual output
                    for (int i = 0; i < Owner.VisualOutput.Count; i++)
                    {
                        Owner.VisualOutput.Host[i] = 0.0f;
                    }

                    if (Owner.HighlightPointInput != null && Owner.HighlightPointInput.Count >= 2)
                    {
                        Owner.HighlightPointInput.SafeCopyToHost();

                        for (int i = 0; i + 2 <= Owner.HighlightPointInput.Count; i += 2)
                        {
                            int highlightX = (int)Owner.HighlightPointInput.Host[i+0];
                            int highlightY = (int)Owner.HighlightPointInput.Host[i+1];
                            int highlightRadius = 10;

                            DrawLine(highlightX + highlightRadius / 2, highlightY, highlightX - highlightRadius / 2, highlightY, Owner.VisualOutput.Host, Owner.WORLD_WIDTH, Owner.WORLD_HEIGHT);
                            DrawLine(highlightX, highlightY + highlightRadius / 2, highlightX, highlightY - highlightRadius / 2, Owner.VisualOutput.Host, Owner.WORLD_WIDTH, Owner.WORLD_HEIGHT);
                        }
                    }

                    //draw joints
                    for (int i = 0; i < Owner.JOINTS; i++)
                    {
                        DrawCircle(Owner.JointPosition[i].X, Owner.JointPosition[i].Y, 2, Owner.VisualOutput.Host, Owner.WORLD_WIDTH, Owner.WORLD_HEIGHT);
                    }

                    //draw bones
                    for (int i = 1; i < Owner.JOINTS + 1; i++)
                    {
                        DrawLine(Owner.JointPosition[i - 1].X, Owner.JointPosition[i - 1].Y, Owner.JointPosition[i].X, Owner.JointPosition[i].Y, Owner.VisualOutput.Host, Owner.WORLD_WIDTH, Owner.WORLD_HEIGHT);
                    }

                    //draw reachable
                    for (int i = 0; i < Owner.VisualOutput.Count; i++)
                    {
                        Owner.VisualOutput.Host[i] += Owner.Reach.Host[i];
                    }

                    if (HIGHLIGHT_X >= 0 && HIGHLIGHT_X < Owner.WORLD_WIDTH && HIGHLIGHT_Y >= 0 && HIGHLIGHT_Y < Owner.WORLD_HEIGHT)
                    {
                        Owner.VisualOutput.Host[HIGHLIGHT_Y * Owner.WORLD_WIDTH + HIGHLIGHT_X] = -1.0f;
                    }
                }

                Owner.EndPosition.Host[0] = Owner.JointPosition[Owner.JOINTS].X;
                Owner.EndPosition.Host[1] = Owner.JointPosition[Owner.JOINTS].Y;
                Owner.EndPosition.SafeCopyToDevice();

                for (int i = 0; i < Owner.JOINTS; i++)
                {
                    Owner.JointsPositionOutput.Host[2 * i] = Owner.JointPosition[i].X;
                    Owner.JointsPositionOutput.Host[2 * i + 1] = Owner.JointPosition[i].Y;
                }
                Owner.JointsPositionOutput.SafeCopyToDevice();

                Owner.VisualOutput.SafeCopyToDevice();
                Owner.MusclesLengthOutput.SafeCopyToDevice();

                //draw virtual arm
                if (UPDATE_VISUAL_OUTPUT && Owner.VirtualMusclesLengthInput != null && Owner.VirtualMusclesLengthInput.Count == Owner.JOINTS)
                {
                    Owner.VirtualMusclesLengthInput.SafeCopyToHost();
                    float virtualRotation = 90; //sum of rotations of all previous joins to calculate real rotation

                    Owner.VirtualJointPosition[0].X = 2 * Owner.WORLD_WIDTH / 3;;
                    Owner.VirtualJointPosition[0].Y = Owner.WORLD_HEIGHT / 2;

                    for (int i = 1; i < Owner.JOINTS + 1; i++)
                    {
                        if (Owner.JointMinRotation[i] <= Owner.JointMaxRotation[i]) //continuous interval
                        {
                            virtualRotation += Owner.JointMinRotation[i] + (1-Owner.VirtualMusclesLengthInput.Host[i - 1]) * (Owner.JointMaxRotation[i] - Owner.JointMinRotation[i]);
                        }
                        else
                        {
                            virtualRotation += Owner.JointMinRotation[i] + (1-Owner.VirtualMusclesLengthInput.Host[i - 1]) * (Owner.JointMaxRotation[i] + (360 - Owner.JointMinRotation[i]));
                        }

                        Owner.VirtualJointPosition[i].X = Owner.VirtualJointPosition[i - 1].X + (int)(Math.Cos(virtualRotation * (Math.PI / 180)) * Owner.BoneLength[i - 1]);
                        Owner.VirtualJointPosition[i].Y = Owner.VirtualJointPosition[i - 1].Y + (int)(Math.Sin(virtualRotation * (Math.PI / 180)) * Owner.BoneLength[i - 1]);
                    }

                    //clear whole visual output
                    for (int i = 0; i < Owner.VirtualOutput.Count; i++)
                    {
                        Owner.VirtualOutput.Host[i] = 0.0f;
                    }

                    //draw joints
                    for (int i = 0; i < Owner.JOINTS; i++)
                    {
                        DrawCircle(Owner.VirtualJointPosition[i].X, Owner.VirtualJointPosition[i].Y, 2, Owner.VirtualOutput.Host, Owner.WORLD_WIDTH, Owner.WORLD_HEIGHT);
                    }

                    //draw bones
                    for (int i = 1; i < Owner.JOINTS + 1; i++)
                    {
                        DrawLine(Owner.VirtualJointPosition[i - 1].X, Owner.VirtualJointPosition[i - 1].Y, Owner.VirtualJointPosition[i].X, Owner.VirtualJointPosition[i].Y, Owner.VirtualOutput.Host, Owner.WORLD_WIDTH, Owner.WORLD_HEIGHT);
                    }

                    Owner.VirtualOutput.SafeCopyToDevice();
                }
            }

            #region Drawing Utils
            private static void Swap<T>(ref T lhs, ref T rhs) { T temp; temp = lhs; lhs = rhs; rhs = temp; }

            private static void DrawLine(int x0, int y0, int x1, int y1, float[] output, int width, int height)
            {
                if (x0 <= -32768 || y0 <= -32768 || x1 <= -32768 || y1 <= -32768) return;
                bool steep = Math.Abs(y1 - y0) > Math.Abs(x1 - x0);
                if (steep) { Swap<int>(ref x0, ref y0); Swap<int>(ref x1, ref y1); }
                if (x0 > x1) { Swap<int>(ref x0, ref x1); Swap<int>(ref y0, ref y1); }
                int dX = (x1 - x0), dY = Math.Abs(y1 - y0), err = (dX / 2), ystep = (y0 < y1 ? 1 : -1), y = y0;

                for (int x = x0; x <= x1; ++x)
                {
                    if (steep)
                    {
                        if (y >= 0 && y < width && x >= 0 && x < height)
                            output[x * width + y] = 1.0f;
                    }
                    else
                    {
                        if (x >= 0 && x < width && y >= 0 && y < height)
                            output[y * width + x] = 1.0f;
                    }

                    err = err - dY;
                    if (err < 0) { y += ystep; err += dX; }
                }
            }


            private static void DrawCircle(int centerX, int centerY, int radius, float[] output, int width, int height)
            {
                for (int x = centerX - radius; x <= centerX + radius; x++)
                {
                    for (int y = centerY - radius; y <= centerY + radius; y++)
                    {
                        if (Math.Sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY)) <= radius)
                        {
                            if (x >= 0 && x < width && y >= 0 && y < height)
                            {
                                output[y * width + x] = 1.0f;
                            }
                        }
                    }
                }
            }
            #endregion
        }


        public override void UpdateMemoryBlocks()
        {
            VisualOutput.Count = WORLD_WIDTH * WORLD_HEIGHT;
            VisualOutput.ColumnHint = WORLD_WIDTH;
            Reach.Count = WORLD_WIDTH * WORLD_HEIGHT;
            EndPosition.Count = 2;

            MusclesLengthOutput.Count = JOINTS;

            VirtualOutput.Count = WORLD_WIDTH * WORLD_HEIGHT;
            VirtualOutput.ColumnHint = WORLD_WIDTH;

            JointsPositionOutput.Count = JOINTS * 2;
            JointsPositionOutput.ColumnHint = 2;
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(JOINTS <= 8, this, "Maximum number of joints is 8!");
            validator.AssertError(MusclesInput != null && MusclesInput.Count == JOINTS, this, "Muscles input size must be equal to number of joints!");
        }
        
    }
}
