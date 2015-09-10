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
    /// <meta>df</meta>
    /// <status>WIP</status>
    /// <summary>Simple 2D topview world with agent and target.</summary>
    /// <description>Agent and target positions are available. Agent movement can be controlled.</description>
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

            Bitmaps.Count += LoadAndGetBitmapSize(@"res\gridworld\agent.png");
            Bitmaps.Count += LoadAndGetBitmapSize(@"res\gridworld\lightsOn.png");

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
                    Bitmap bitmap = (Bitmap)Image.FromFile(path, true);
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

                Bitmap bitmap = Owner.m_bitmapTable[@"res\gridworld\agent.png"];

                MyGameObject agent = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[@"res\gridworld\lightsOn.png"];

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

        public class MyUpdateTask : MyTask<My2DAgentWorld>
        {            
            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 1.0f), YAXElementFor("Structure")]
            public float REACH_TARGET { get; set; }            
            
            private Random m_random = new Random();

            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                Owner.Event.Host[0] = 0;

                MyGameObject agent = Owner.m_gameObjects[0];
                MyGameObject target = Owner.m_gameObjects[1];

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

                target.position.x = (Owner.DISPLAY_WIDTH - target.pixelSize.x) * (float)m_random.NextDouble();
                target.position.y = (Owner.DISPLAY_HEIGHT - target.pixelSize.y) * (float)m_random.NextDouble();
            }

            private const float AGENT_VELOCITY = 4.0f;
            private const float DELTA_T = 1.0f;
            private const float TARGET_RADIUS = 15.0f;
            private const int RESET_COUNTDOWN_STEPS = 5;

            private int m_resetCountDown;

            private void ResolveAgentEvents(MyGameObject agent, MyGameObject target)
            {
                float2 futurePos = agent.position + agent.velocity * DELTA_T;

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

                if (relPos.x * relPos.x + relPos.y * relPos.y < TARGET_RADIUS * TARGET_RADIUS)                
                {
                    Owner.Event.Host[0] += REACH_TARGET;

                    target.position.x = (Owner.DISPLAY_WIDTH - target.pixelSize.x) * (float)m_random.NextDouble();
                    target.position.y = (Owner.DISPLAY_HEIGHT - target.pixelSize.y) * (float)m_random.NextDouble();

                    if (m_resetCountDown == 0)
                    {
                        m_resetCountDown = RESET_COUNTDOWN_STEPS;
                    }
                }

                agent.position = agent.position + agent.velocity * DELTA_T;                                
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
                    agent.velocity.y = -AGENT_VELOCITY;
                }
                else if (maxAction >= 6)
                {
                    agent.velocity.y = AGENT_VELOCITY;
                }

                if (maxAction % 3 == 0)
                {
                    agent.velocity.x = -AGENT_VELOCITY;
                }
                else if (maxAction % 3 == 2)
                {
                    agent.velocity.x = AGENT_VELOCITY;
                }                               
            }            
        }
    }
}
