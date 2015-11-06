using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using YAXLib;

namespace GoodAI.Modules.GameBoy
{
    /// <author>GoodAI</author>
    /// <meta>df,jv</meta>
    /// <status>Working</status>
    /// <summary>Simple 2D topview world with agent and target.</summary>
    /// <description>Agent and target positions are available. Agent moves continuously in the 2D environment. 
    /// Movement is controlled by 9 values (8 directions + no operation), which change velocity in the corresponding direction. 
    /// The goal is to reach the target, if target is reached, the event signal is sent and goal is placed onto new randomly generated position.
    /// </description>
    public class My2DAgentWorld : MyWorld
    {
        public class MyGameObject
        {
            public float2 position;
            public int2 pixelSize;
            public float2 velocity;
            public CUdeviceptr bitmap;
        };

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Controls
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Visual
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Event
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> AgentPosition
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> TargetPosition
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 160)]
        public int DISPLAY_WIDTH { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 160)]
        public int DISPLAY_HEIGHT { get; set; }

        public MyMemoryBlock<float> Bitmaps { get; private set; }

        private Dictionary<string, Bitmap> m_bitmapTable = new Dictionary<string, Bitmap>();
        private string m_errorMessage;

        private List<MyGameObject> m_gameObjects;

        public override void UpdateMemoryBlocks()
        {
            Visual.Count = DISPLAY_WIDTH * DISPLAY_HEIGHT;
            Visual.ColumnHint = DISPLAY_WIDTH;

            Bitmaps.Count = 0;

            Bitmaps.Count += LoadAndGetBitmapSize(@"res\gridworld3\agent.png");
            Bitmaps.Count += LoadAndGetBitmapSize(@"res\gridworld3\lightsOn.png");

            AgentPosition.Count = 2;
            TargetPosition.Count = 2;

            Event.Count = 1;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            if (Controls != null)
            {
                validator.AssertError(Controls.Count >= 9, this, "Not enough controls");
            }

            validator.AssertError(Bitmaps.Count != 0, this, "Node cannot be executed. Some resources are missing: " + m_errorMessage);
        }

        private int LoadAndGetBitmapSize(string path)
        {
            if (!m_bitmapTable.ContainsKey(path))
            {
                try
                {
                    Bitmap bitmap = (Bitmap)Image.FromFile(MyResources.GetMyAssemblyPath() + "\\" + path, true);
                    m_bitmapTable[path] = bitmap;
                    return bitmap.Width * bitmap.Height;
                }
                catch (Exception ex)
                {
                    m_errorMessage = ex.Message;
                    return 0;
                }
            }
            else return m_bitmapTable[path].Width * m_bitmapTable[path].Height;
        }

        public MyInitTask InitGameTask { get; private set; }
        public MyUpdateTask UpdateTask { get; private set; }
        public MyRenderTask RenderGameTask { get; private set; }

        /// <summary>
        /// Initialize the simulation, load bitmaps.
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<My2DAgentWorld>
        {
            private int FillWithChannelFromBitmap(Bitmap bitmap, int channel, float[] buffer, int offset)
            {
                BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

                byte[] pixels = new byte[bitmapData.Stride];

                int bytesPerPixel = bitmapData.Stride / bitmapData.Width;

                for (int i = 0; i < bitmap.Height; i++)
                {
                    Marshal.Copy(bitmapData.Scan0, pixels, 0, pixels.Length);
                    bitmapData.Scan0 += bitmapData.Stride;

                    for (int j = 0; j < bitmap.Width; j++)
                    {
                        buffer[i * bitmap.Width + j + offset] = pixels[j * bytesPerPixel + channel] / 255.0f;
                    }
                }
                bitmap.UnlockBits(bitmapData);
                return bitmap.Width * bitmap.Height;
            }

            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                int offset = 0;
                Owner.m_gameObjects = new List<MyGameObject>();
                CudaDeviceVariable<float> devBitmaps = Owner.Bitmaps.GetDevice(Owner);

                Bitmap bitmap = Owner.m_bitmapTable[@"res\gridworld3\agent.png"];

                MyGameObject agent = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[@"res\gridworld3\lightsOn.png"];

                MyGameObject target = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                Owner.m_gameObjects.Add(agent);
                Owner.m_gameObjects.Add(target);

                Owner.Bitmaps.SafeCopyToDevice();

                Owner.UpdateTask.ResetGame();
            }
        }

        /// <summary>
        /// Render the world to the visual output.
        /// </summary>
        public class MyRenderTask : MyTask<My2DAgentWorld>
        {
            private MyCudaKernel m_kernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "DrawSpriteKernel");
            }

            public override void Execute()
            {
                Owner.Visual.Fill(1.0f);

                for (int i = 0; i < Owner.m_gameObjects.Count; i++)
                {
                    MyGameObject g = Owner.m_gameObjects[i];

                    m_kernel.SetupExecution(g.pixelSize.x * g.pixelSize.y);
                    m_kernel.Run(Owner.Visual, Owner.DISPLAY_WIDTH, Owner.DISPLAY_HEIGHT, g.bitmap, g.position, g.pixelSize);
                }
            }
        }

        /// <summary>
        /// Receive agent's action, apply the world rules and publish new state.
        /// 
        /// <description>
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li> <b>ReachTargetSignal: </b>This value is set to the Event output for one time step, if the target is reached.</li>
        ///     <li> <b>AgentVelocity: </b>How fast the agent moves.</li>
        ///     <li> <b>DeltaT: </b>Time shift at each simulation step.</li>
        ///     <li> <b>TargetRadius: </b>Radius in which the target is considered as reached.</li>
        ///     <li> <b>ResetCountouwnSteps: </b>How many steps after reaching the target to wait before generating new target.</li>
        ///     <li> <b>ForcePosition: </b>After reaching the target, new position is generated. If true, position is forced to be constant.</li>
        /// </ul>
        /// 
        /// </description>
        /// </summary>
        public class MyUpdateTask : MyTask<My2DAgentWorld>
        {
            #region parameters

            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 1.0f), YAXElementFor("Structure"),
            Description("This value is set to the Event output for one time step, if the target is reached.")]
            public float ReachTargetSignal
            {
                get { return m_reachTarget; }
                set { m_reachTarget = value; }
            }
            private float m_reachTarget;

            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 4.0f), YAXElementFor("Structure"),
            Description("How fast the agent moves.")]
            public float AgentVelocity
            {
                get { return m_agentVelocity; }
                set
                {
                    if (value > 0)
                    {
                        m_agentVelocity = value;
                    }
                }
            }
            private float m_agentVelocity;


            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 1.0f), YAXElementFor("Structure"),
            Description("Time shift at each simulation step.")]
            public float DeltaT
            {
                get { return m_deltaT; }
                set
                {
                    if (value > 0)
                    {
                        m_deltaT = value;
                    }
                }
            }
            private float m_deltaT;


            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 15.0f), YAXElementFor("Structure"),
            Description("Radius in which the target is considered as reached.")]
            public float TargetRadius
            {
                get { return m_targetRadius; }
                set
                {
                    if (value > 0)
                    {
                        m_targetRadius = value;
                    }
                }
            }
            private float m_targetRadius;


            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 5), YAXElementFor("Structure"),
            Description("How many steps after reaching the target to wait before generating new target.")]
            public int ResetCountouwnSteps
            {
                get { return m_resetCountdownSteps; }
                set
                {
                    if (value >= 0)
                    {
                        m_resetCountdownSteps = value;
                    }
                }
            }
            private int m_resetCountdownSteps;


            [MyBrowsable, Category("Force Position")]
            [YAXSerializableField(DefaultValue = false), YAXElementFor("Force Position"),
            Description("Force position to a given coordinates?.")]
            public bool ForcePosition { get; set; }

            [MyBrowsable, Category("Force Position")]
            [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Force Position"),
            Description("Forced X position of the reward")]
            public float XPosition {
                get {
                    return m_xPos;
                }
                set{
                    if (value >= 0 && value <= 1)
                    {
                        m_xPos = value;
                    }
                }
            }
            private float m_xPos;

            [MyBrowsable, Category("Force Position")]
            [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Force Position"),
            Description("Forced Y position of the reward")]
            public float YPosition
            {
                get
                {
                    return m_yPos;
                }
                set
                {
                    if (value >= 0 && value <= 1)
                    {
                        m_yPos = value;
                    }
                }
            }
            private float m_yPos;

            #endregion

            private Random m_random = new Random();

            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                MyGameObject target = Owner.m_gameObjects[1];
                if (ForcePosition)
                {
                    SetTargetPosition(target);
                }
                Owner.Event.Host[0] = 0;

                MyGameObject agent = Owner.m_gameObjects[0];

                Owner.Controls.SafeCopyToHost();

                if (m_resetCountDown > 0)
                {
                    m_resetCountDown--;

                    if (m_resetCountDown == 0)
                    {
                        ResetAgent();
                        //ResetGame();
                    }
                }

                ApplyControl(agent);
                ResolveAgentEvents(agent, target);

                Owner.AgentPosition.Host[0] = agent.position.x + agent.pixelSize.x * 0.5f;
                Owner.AgentPosition.Host[1] = agent.position.y + agent.pixelSize.y * 0.5f;

                Owner.TargetPosition.Host[0] = target.position.x + target.pixelSize.x * 0.5f;
                Owner.TargetPosition.Host[1] = target.position.y + target.pixelSize.y * 0.5f;

                Owner.AgentPosition.SafeCopyToDevice();
                Owner.TargetPosition.SafeCopyToDevice();

                Owner.Event.SafeCopyToDevice();
            }

            public void ResetGame()
            {
                ResetAgent();
                ResetTarget();
            }

            private void ResetAgent()
            {
                MyGameObject agent = Owner.m_gameObjects[0];

                agent.position.x = (Owner.DISPLAY_WIDTH - agent.pixelSize.x) * (float)m_random.NextDouble();
                agent.position.y = (Owner.DISPLAY_HEIGHT - agent.pixelSize.y) * (float)m_random.NextDouble();

                agent.velocity.x = 0;
                agent.velocity.y = 0;
            }

            private void ResetTarget()
            {
                MyGameObject target = Owner.m_gameObjects[1];
                SetTargetPosition(target);
            }

            private int m_resetCountDown;

            private void ResolveAgentEvents(MyGameObject agent, MyGameObject target)
            {
                float2 futurePos = agent.position + agent.velocity * DeltaT;

                //topSide
                if (futurePos.y < 0 && agent.velocity.y < 0)
                {
                    agent.velocity.y = 0;
                }
                //bottomSide
                if (futurePos.y + agent.pixelSize.y > Owner.DISPLAY_HEIGHT && agent.velocity.y > 0)
                {
                    agent.velocity.y = 0;
                }

                //leftSide
                if (futurePos.x < 0 && agent.velocity.x < 0)
                {
                    agent.velocity.x = 0;
                }
                //rightSide
                if (futurePos.x + agent.pixelSize.x > Owner.DISPLAY_WIDTH && agent.velocity.x > 0)
                {
                    agent.velocity.x = 0;
                }

                //target
                float2 relPos = agent.position - target.position;

                if (relPos.x * relPos.x + relPos.y * relPos.y < TargetRadius * TargetRadius)
                {
                    Owner.Event.Host[0] += ReachTargetSignal;
                    SetTargetPosition(target);
                    
                    if (m_resetCountDown == 0)
                    {
                        m_resetCountDown = ResetCountouwnSteps;
                    }
                }

                agent.position = agent.position + agent.velocity * DeltaT;
            }

            private void SetTargetPosition(MyGameObject target)
            {
                if (ForcePosition)
                {
                    target.position.x = (Owner.DISPLAY_WIDTH - target.pixelSize.x) * XPosition;
                    target.position.y = (Owner.DISPLAY_HEIGHT - target.pixelSize.y) * YPosition;
                }
                else
                {
                    target.position.x = (Owner.DISPLAY_WIDTH - target.pixelSize.x) * (float)m_random.NextDouble();
                    target.position.y = (Owner.DISPLAY_HEIGHT - target.pixelSize.y) * (float)m_random.NextDouble();
                }
            }

            private void ApplyControl(MyGameObject agent)
            {
                agent.velocity.x = 0;
                agent.velocity.y = 0;

                int maxAction = 4;
                float maxActionValue = 0;

                for (int i = 0; i < 9; i++)
                {
                    if (maxActionValue < Owner.Controls.Host[i])
                    {
                        maxActionValue = Owner.Controls.Host[i];
                        maxAction = i;
                    }
                }

                if (maxAction < 3)
                {
                    agent.velocity.y = -AgentVelocity;
                }
                else if (maxAction >= 6)
                {
                    agent.velocity.y = AgentVelocity;
                }

                if (maxAction % 3 == 0)
                {
                    agent.velocity.x = -AgentVelocity;
                }
                else if (maxAction % 3 == 2)
                {
                    agent.velocity.x = AgentVelocity;
                }
            }
        }
    }
}
