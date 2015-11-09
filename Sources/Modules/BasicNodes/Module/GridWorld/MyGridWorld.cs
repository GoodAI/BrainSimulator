using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
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

namespace GoodAI.Modules.GridWorld
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>A simple discrete simulator with maze and controllable objects.</summary>
    /// <description>
    /// 
    /// A discrete world with one agent, the agent can:
    /// <ul>
    ///     <li>Do nothiong (action 0)</li>
    ///     <li>Move in 4 directions (actions 1,2,3,4)</li>
    ///     <li>and control controllable objects (action 5).</li>
    /// </ul>
    /// 
    /// The world is composed of:
    /// <ul>
    ///     <li>an agent</li>
    ///     <li>free tales</li>
    ///     <li>obstacles</li>
    ///     <li>two types of static objects: controllable (door switch, lights switch) and controlled objects (door, lights)</li>
    /// </ul>
    /// 
    /// 
    /// 
    /// <h3>Outputs</h3>
    /// <ul>
    ///     <li> <b>Global:</b> publishes all information about the world, that is:
    /// agent's position and (changeable) properties of all objects with their positions.</li>
    ///     <li> <b>Variables:</b> publishes only all variables in the world (typically omits positions
    /// of static objects), for testing purposes.</li>
    /// <li> <b>AgentPosX:</b> Agent's current position on the X axis.</li>
    /// <li> <b>AgentPosY:</b> Agent's current position on the Y axis.</li>
    /// <li> <b>Visual:</b> bitmap representing the current world state.</li>
    /// <li> <b>EgocentricVisual:</b> egocentric view of the agent with predefined size</li>
    /// <li> <b>MapSizeOutput:</b> publishes the following vector of information about wolrd size: [maxX, maxY, 1/maxX, 1/maxY].</li>
    /// </ul>
    /// 
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li> <b>Action:</b> Vector indicating the selected action. The index of maximum value is evaluated as a selected action. 
    ///     Actions are in the following order: NOOP,LEFT,RIGHT,UP,DOWN,PRESS.</li>
    /// </ul>
    /// 
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li><b>ShowInEgocentricView: </b>show the agent in egocentric view?</li>
    ///     <li><b>EgocentricViewLimit: </b>size of the egocentric visual</li>
    ///     <li><b>WorldBoundsValue: </b>used for boundaries in the egocentric view</li>
    ///     <li><b>USED_MAP: </b>chooses one of predefined maps</li>
    ///     <li><b>TEXTURE: </b>different textures for the visual representation</li>
    /// </ul>
    /// 
    /// </description>
    public class MyGridWorld : MyWorld
    {
        [MyInputBlock]
        public MyMemoryBlock<float> ActionInput
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
        public MyMemoryBlock<float> AgentPosX
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> AgentPosY
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> GlobalOutput
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> VariablesOutput
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        [MyOutputBlock(5)]
        public MyMemoryBlock<float> EgocentricVisual
        {
            get { return GetOutput(5); }
            set { SetOutput(5, value); }
        }

        [MyOutputBlock(6)]
        public MyMemoryBlock<float> MapSizeOutput
        {
            get { return GetOutput(6); }
            set { SetOutput(6, value); }
        }

        [MyBrowsable, Category("Maps")]
        [YAXSerializableField(DefaultValue = MyCustomMaps.mapB)]
        public MyCustomMaps USED_MAP
        {
            get
            {
                return mc;
            }
            set
            {
                mc = value;
            }
        }

        [MyBrowsable, Category("Texture Sets"), Description("Choose which texture set to use for rendering the world.")]
        [YAXSerializableField(DefaultValue = MyTextureSets.textureB)]
        public MyTextureSets TEXTURE
        {
            get
            {
                return ts;
            }
            set
            {
                ts = value;
            }
        }

        MyCustomMaps mc;
        public enum MyCustomMaps
        {
            mapA = 0,
            mapB = 1,
            mapC = 2,
            mapD = 3,
            mapE = 4,
            mapF = 5,
            mapG = 6,
            mapH = 7,
            mapI = 8,
            mapJ = 9,
            mapK = 10,
            mapL = 11
        };

        MyTextureSets ts;
        public enum MyTextureSets
        {
            textureA = 0,
            textureB = 1,
            textureC = 2,
            textureD = 3,
            textureE = 4
        };

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 16)]
        public int RES { get; private set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 160)]
        public int VISIBLE_HEIGHT { get; private set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 160)]
        public int VISIBLE_WIDTH { get; private set; }

        [MyBrowsable, Category("Agent")]
        [YAXSerializableField(DefaultValue = true)]
        public bool ShowInEgocentricView { get; set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 2)]
        public int EgocentricViewLimit { get; set; }

        [MyBrowsable, Category("Constants")]
        [YAXSerializableField(DefaultValue = 0)]
        public float WorldBoundsValue { get; set; }

        private bool m_performingMovement;
        private int m_movementCooldown;
        public AGENT_ACTIONS LastAction { get; private set; }

        private float AGENT_WEIGHT = 0.5f;

        public class MyNewStateSignal : MySignal { }

        public MyNewStateSignal HasNewState { get; private set; }

        private Dictionary<string, Bitmap> m_bitmapTable = new Dictionary<string, Bitmap>();
        public MyMemoryBlock<float> Bitmaps { get; private set; }
        private string m_errorMessage;

        public IWorldParser World { get; private set; }
        public IWorldEngine Engine { get; private set; }

        private MyGraphicsPrototype m_tale_empty_g, m_tale_obstacle_g, m_agent_g;

        public class MyGraphicsPrototype
        {
            public int2 PixelSize;
            public CUdeviceptr Bitmap;
        }

        // definition of the map for GPU rendering, tales can be: empty/obstacle (empty ones may contain other objects)
        public MyMemoryBlock<int> MapTales { get; private set; }

        private string TEXTURE_SET = @"res\gridworld2\";
        public override void UpdateMemoryBlocks()
        {
            Bitmaps.Count = 0;

            switch (TEXTURE)
            {
                case MyTextureSets.textureA:
                    TEXTURE_SET = @"res\gridworld\";
                    break;
                case MyTextureSets.textureB:
                    TEXTURE_SET = @"res\gridworld2\";
                    break;
                case MyTextureSets.textureC:
                    TEXTURE_SET = @"res\gridworld3\";
                    break;
                case MyTextureSets.textureD:
                    TEXTURE_SET = @"res\gridworld4\";
                    break;
                case MyTextureSets.textureE:
                    TEXTURE_SET = @"res\gridworld5\";
                    break;
            }

            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleEmpty.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "taleObstacle.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "agent.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "doorClosed.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "doorOpened.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "doorControl.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "doorControlOff.png");

            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "lightsControl.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "lightsControlOff.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "lightsOn.png");
            Bitmaps.Count += LoadAndGetBitmapSize(TEXTURE_SET + "lightsOff.png");

            AgentPosX.Count = 1;
            AgentPosY.Count = 1;

            // user can choose from the following maps
            switch (USED_MAP)
            {
                case MyCustomMaps.mapA:
                    World = new MyMapA();
                    break;
                case MyCustomMaps.mapB:
                    World = new MyMapB();
                    break;
                case MyCustomMaps.mapC:
                    World = new MyMapC();
                    break;
                case MyCustomMaps.mapD:
                    World = new MyMapD();
                    break;
                case MyCustomMaps.mapE:
                    World = new MyMapE();
                    break;
                case MyCustomMaps.mapF:
                    World = new MyMapF();
                    break;
                case MyCustomMaps.mapG:
                    World = new MyMapG();
                    break;
                case MyCustomMaps.mapH:
                    World = new MyMapH();
                    break;
                case MyCustomMaps.mapI:
                    World = new MyMapI();
                    break;
                case MyCustomMaps.mapJ:
                    World = new MyMapJ();
                    break;
                case MyCustomMaps.mapK:
                    World = new MyMapK();
                    break;
                case MyCustomMaps.mapL:
                    World = new MyMapL();
                    break;
                default:
                    World = new MyMapA();
                    break;
            }
            // instantiate the world simulation engine
            Engine = new SimpleGridWorldEngine(World);

            MapTales.Count = World.GetWidth() * World.GetHeight();
            MapTales.ColumnHint = World.GetWidth();

            GlobalOutput.Count = Engine.GetGlobalOutputDataSize();
            VariablesOutput.Count = Engine.GetVariablesOutputDataSize();

            VISIBLE_HEIGHT = World.GetHeight() * RES;
            VISIBLE_WIDTH = World.GetWidth() * RES;

            Visual.Count = World.GetHeight() * World.GetWidth() * RES * RES;
            Visual.ColumnHint = World.GetWidth() * RES;

            EgocentricVisual.Count = RES * RES * (EgocentricViewLimit * 2 + 1) * (EgocentricViewLimit * 2 + 1);
            EgocentricVisual.ColumnHint = (int)Math.Sqrt(EgocentricVisual.Count);

            World.GetAgent().SetWeight(AGENT_WEIGHT);

            MapSizeOutput.Count = 4;
        }

        public void PublishWorldSize()
        {
            if (World.GetWidth() == 0 || World.GetHeight() == 0)
            {
                MyLog.DEBUG.WriteLine("WARNING: unexpected size of the world: [" +
                    World.GetWidth() + "," + World.GetHeight() + "]");
            }
            MapSizeOutput.SafeCopyToHost();

            MapSizeOutput.Host[0] = World.GetWidth() - 1;
            MapSizeOutput.Host[1] = World.GetHeight() - 1;
            MapSizeOutput.Host[2] = 1 / ((float)World.GetWidth() - 1);
            MapSizeOutput.Host[3] = 1 / ((float)World.GetHeight() - 1);
            MapSizeOutput.SafeCopyToDevice();
        }

        // @see MyCustomPongWorld
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

        // @see MyCustomPongWorld
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

        public override void Validate(MyValidator validator)
        {
        }

        public MyResetAgentTask ResetAgentTask { get; private set; }
        public MyInitTask InitGameTask { get; private set; }
        public MyUpdateTask UpdateTask { get; private set; }
        public MyRenderTask RenderGameTask { get; private set; }

        /// <summary>
        /// Initialize the world (load graphics etc.).
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MyGridWorld>
        {
            public override void Init(int nGPU)
            {
            }

            public override void Execute()
            {
                // load bitmaps and pas them to the game objects
                int offset = 0;
                CudaDeviceVariable<float> devBitmaps = Owner.Bitmaps.GetDevice(Owner);

                Bitmap bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "taleObstacle.png"];
                Owner.m_tale_obstacle_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "taleEmpty.png"];
                Owner.m_tale_empty_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "agent.png"];
                Owner.m_agent_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "doorOpened.png"];
                MyGraphicsPrototype doorOpened_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "doorClosed.png"];
                MyGraphicsPrototype doorClosed_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "doorControl.png"];
                MyGraphicsPrototype doorControl_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "doorControlOff.png"];
                MyGraphicsPrototype doorControlOff_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "lightsControl.png"];
                MyGraphicsPrototype lightsControl_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "lightsControlOff.png"];
                MyGraphicsPrototype lightsControlOff_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);


                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "lightsOn.png"];
                MyGraphicsPrototype lightsOn_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[Owner.TEXTURE_SET + "lightsOff.png"];
                MyGraphicsPrototype lightsOff_g = new MyGraphicsPrototype()
                {
                    PixelSize = new int2(bitmap.Width, bitmap.Height),
                    Bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                // parse the map and instantiate all parts of the world
                Owner.World.registerGraphics(
                    Owner.m_tale_empty_g, Owner.m_tale_obstacle_g, Owner.m_agent_g,
                    doorOpened_g, doorClosed_g, doorControl_g, doorControlOff_g,
                    lightsControl_g, lightsControlOff_g, lightsOff_g, lightsOn_g);

                Owner.AgentPosX.Host[0] = Owner.World.GetAgent().GetPosition().x;
                Owner.AgentPosY.Host[0] = Owner.World.GetAgent().GetPosition().y;

                Owner.AgentPosX.SafeCopyToDevice();
                Owner.AgentPosY.SafeCopyToDevice();
                Owner.Bitmaps.SafeCopyToDevice();
                Array.Copy(Owner.World.GetArray(), Owner.MapTales.Host, Owner.MapTales.Count);
                Owner.MapTales.SafeCopyToDevice();

                Owner.PublishWorldSize();
            }


            // @see MyCustomPongWorld
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
        }

        /// <summary>
        /// Renders the visible area, not needed for simulation.
        /// </summary>
        public class MyRenderTask : MyTask<MyGridWorld>
        {
            private MyCudaKernel m_drawTalesKernel;
            private MyCudaKernel m_drawObjKernel;
            private MyCudaKernel m_drawFreeObjKernel;

            private MyCudaKernel m_copyKernel;

            private int2 m_agentPosition;

            public override void Init(int nGPU)
            {
                m_drawObjKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\GridWorld", "DrawObjectKernel");
                m_drawFreeObjKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\GridWorld", "DrawFreeObjectKernel");

                m_drawTalesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\GridWorld", "DrawTalesKernel");

                m_copyKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CopyRectangleKernel", "CopyRectangleCheckBoundsKernel");
            }

            public override void Execute()
            {
                // draw the visible area of the map (empty and obstacle tales)
                MyGraphicsPrototype taleEmpty = Owner.m_tale_empty_g;
                MyGraphicsPrototype taleObstacle = Owner.m_tale_obstacle_g;

                m_drawTalesKernel.SetupExecution(Owner.VISIBLE_WIDTH * Owner.VISIBLE_HEIGHT);
                m_drawTalesKernel.Run(Owner.Visual, Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT,
                    Owner.MapTales, Owner.World.GetWidth(), Owner.World.GetHeight(),
                    taleEmpty.Bitmap, taleObstacle.Bitmap, taleEmpty.PixelSize);

                // draw static objects
                MyStaticObject[] objects = Owner.World.GetStaticObjects();
                for (int i = 0; i < objects.Length; i++)
                {
                    MyStaticObject o = objects[i];

                    m_drawObjKernel.SetupExecution(o.Graphics.PixelSize.x * o.Graphics.PixelSize.y);

                    m_drawObjKernel.Run(Owner.Visual, Owner.RES,
                        Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT,
                        o.Graphics.Bitmap, o.GetPosition(), o.Graphics.PixelSize);
                }

                // draw the agent
                MyMovingObject agent = Owner.World.GetAgent();
                MyGraphicsPrototype agent_g = agent.Graphics;

                m_agentPosition.x = agent.GetPosition().x * Owner.RES;
                m_agentPosition.y = (agent.GetPosition().y + 1) * Owner.RES - 1;

                if (!Owner.ShowInEgocentricView)
                {
                    this.DrawEgocentric();
                }

                if (Owner.UpdateTask.ContinuousMovement && Owner.m_performingMovement)
                {
                    switch (Owner.LastAction)
                    {
                        case AGENT_ACTIONS.LEFT:
                            {
                                m_agentPosition.x += Owner.m_movementCooldown;
                                break;
                            }
                        case AGENT_ACTIONS.RIGHT:
                            {
                                m_agentPosition.x -= Owner.m_movementCooldown;
                                break;
                            }
                        case AGENT_ACTIONS.UP:
                            {
                                m_agentPosition.y -= Owner.m_movementCooldown;
                                break;
                            }
                        case AGENT_ACTIONS.DOWN:
                            {
                                m_agentPosition.y += Owner.m_movementCooldown;
                                break;
                            }
                    }

                    m_drawFreeObjKernel.SetupExecution(agent_g.PixelSize.x * agent_g.PixelSize.y);
                    m_drawFreeObjKernel.Run(Owner.Visual, Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT, agent_g.Bitmap,
                        m_agentPosition, agent_g.PixelSize);
                }
                else
                {
                    m_drawObjKernel.SetupExecution(agent_g.PixelSize.x * agent_g.PixelSize.y);
                    m_drawObjKernel.Run(Owner.Visual, Owner.RES,
                        Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT,
                        agent_g.Bitmap, agent.GetPosition(), agent_g.PixelSize);
                }

                if (Owner.ShowInEgocentricView)
                {
                    this.DrawEgocentric();
                }

            }

            internal void DrawEgocentric()
            {

                int viewLimit = (Owner.EgocentricViewLimit * 2 + 1) * Owner.RES;

                m_copyKernel.SetupExecution(Owner.EgocentricVisual.Count);
                m_copyKernel.Run(Owner.Visual, 0, Owner.VISIBLE_WIDTH, Owner.VISIBLE_HEIGHT,
                    m_agentPosition.x - Owner.EgocentricViewLimit * Owner.RES,
                    Owner.VISIBLE_HEIGHT - m_agentPosition.y - 1 - Owner.EgocentricViewLimit * Owner.RES, viewLimit, viewLimit,
                    Owner.EgocentricVisual, 0, viewLimit, 0, 0, Owner.WorldBoundsValue);
            }
        }

        /// <summary>
        /// Update the world state based on actions, read the new state and publish it.
        /// 
        /// <description>
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li><b>ForceDoorSwitches: </b>forces the state of door switches in a selected position / agent cannot change it</li>
        ///     <li><b>ForceDoorSwitchesState: </b>defines state of all door switches in the game, if ForceDoorSwitches is enabled</li>
        ///     <li><b>ForceLightSwitches: </b>forces the state of light switches in a selected position / agent cannot change it</li>
        ///     <li><b>ForceLightSwitchesState: </b>defines state of all light switches in the game, if ForceLightSwitches is enabled</li>
        ///     <li><b>LimitFieldOfView: </b>if enabled, only those values of World's output are updated, which correspond to objects that are in the agent's visible area</li>
        ///     <li><b>DoorsAlwaysPassable: </b>if enabled , the agent is always able to go through the door</li>
        ///     <li><b>ContinuousMovement: </b>if enabled , the agent moves in a continuous way</li>
        /// </ul>
        /// </description>
        /// </summary>
        public class MyUpdateTask : MyTask<MyGridWorld>
        {

            [MyBrowsable, Category("Force State Door"),
            Description("Forces the state of door switches in a selected position / agent cannot change it")]
            [YAXSerializableField(DefaultValue = false)]
            public bool ForceDoorSwitches { get; set; }

            [MyBrowsable, Category("Force State Door"),
            Description("Defines state of all door switches in the game, if ForceDoorSwitches is enabled")]
            [YAXSerializableField(DefaultValue = false)]
            public bool ForceDoorSwitchesState { get; set; }

            [MyBrowsable, Category("Force State Lights"),
            Description("Forces the state of light switches in a selected position / agent cannot change it")]
            [YAXSerializableField(DefaultValue = false)]
            public bool ForceLightSwitches { get; set; }

            [MyBrowsable, Category("Force State Lights"),
            Description("Defines state of all light switches in the game, if ForceLightSwitches is enabled")]
            [YAXSerializableField(DefaultValue = false)]
            public bool ForceLightSwitchesState { get; set; }

            [MyBrowsable, Category("Limited Field of View"),
            Description("If enabled, only those values of World's output are updated, which correspond to objects that are in the agent's visible area")]
            [YAXSerializableField(DefaultValue = false)]
            public bool LimitFieldOfView { get; set; }

            [YAXSerializableField(DefaultValue = false)]
            private bool m_contMove;

            [MyBrowsable, Category("Mode"),
            Description("If enabled , the agent is always able to go through the door")]
            public bool DoorsAlwaysPassable
            {
                get { return Owner.World.GetAgent().GetWeight() > 1; }
                set { Owner.World.GetAgent().SetWeight(value ? 1.5f : MyStaticObject.AGENT_W); }
            }

            [MyBrowsable, Category("Mode"),
            Description("If enabled , the agent moves in a continuous way")]
            public bool ContinuousMovement
            {
                get { return m_contMove; }
                set
                {
                    m_contMove = value;
                    ResetMovement();
                }
            }

            public override void Init(int nGPU)
            {
                ResetMovement();
            }

            private void ResetMovement()
            {
                Owner.m_performingMovement = false;
                Owner.m_movementCooldown = 0;
                Owner.LastAction = AGENT_ACTIONS.NOOP;
                Owner.HasNewState.Drop();
            }

            public override void Execute()
            {
                this.updateParams();
                Owner.PublishWorldSize();

                Owner.VariablesOutput.SafeCopyToHost();

                if (ContinuousMovement && Owner.m_performingMovement)
                {
                    Owner.m_movementCooldown--;

                    if (Owner.m_movementCooldown == 0)
                    {
                        Owner.m_performingMovement = false;
                        Owner.HasNewState.Raise();
                    }
                }
                else
                {
                    Owner.HasNewState.Raise();

                    Owner.LastAction = this.decodeAction();
                    Owner.m_performingMovement = Owner.Engine.ResolveAction(Owner.LastAction);

                    if (Owner.m_performingMovement)
                    {
                        Owner.m_movementCooldown = Owner.RES;

                        if (ContinuousMovement)
                        {
                            Owner.HasNewState.Drop();
                        }
                    }

                    float[] outputData;

                    if (LimitFieldOfView)
                    {
                        outputData = Owner.Engine.GetGlobalOutputData();
                    }
                    else
                    {
                        outputData = Owner.Engine.GetGlobalOutputData();

                    }
                    for (int i = 0; i < outputData.Length; i++)
                    {
                        Owner.GlobalOutput.Host[i] = outputData[i];
                    }

                    float[] varData = Owner.Engine.GetVariablesOutputData();

                    for (int i = 0; i < varData.Length; i++)
                    {
                        if (varData[i] != SimpleGridWorldEngine.NOT_VISIBLE)
                        {
                            Owner.VariablesOutput.Host[i] = varData[i];
                        }
                    }

                    Owner.AgentPosY.Host[0] = Owner.World.GetAgent().GetPosition().y;
                    Owner.AgentPosX.Host[0] = Owner.World.GetAgent().GetPosition().x;
                    Owner.AgentPosX.SafeCopyToDevice();
                    Owner.AgentPosY.SafeCopyToDevice();
                    Owner.GlobalOutput.SafeCopyToDevice();
                    Owner.VariablesOutput.SafeCopyToDevice();
                }
            }

            // get action with the max (utility) value
            internal AGENT_ACTIONS decodeAction()
            {
                if (Owner.ActionInput == null)
                {
                    return 0;
                }
                Owner.ActionInput.SafeCopyToHost();

                int result = 0;

                for (int i = 0; i < Owner.ActionInput.Host.Length; i++)
                {
                    if (Owner.ActionInput.Host[result] < Owner.ActionInput.Host[i])
                    {
                        result = i;
                    }
                }
                return (AGENT_ACTIONS)result;
            }

            internal void updateParams()
            {
                Owner.Engine.GetParams().ForceDoorSwitches = ForceDoorSwitches;
                Owner.Engine.GetParams().ForceDoorSwitchesState = ForceDoorSwitchesState;
                Owner.Engine.GetParams().ForceLightSwitchesState = ForceLightSwitchesState;
                Owner.Engine.GetParams().ForceLightSwitches = ForceLightSwitches;
                Owner.Engine.GetParams().ViewLimit = Owner.EgocentricViewLimit;
                Owner.Engine.GetParams().LimitFieldOfView = LimitFieldOfView;
            }
        }

        /// <summary>
        /// Resets the agent position in the world to the specified [x,y] coordinates.
        /// </summary>
        [MyTaskInfo(DesignTime = true)]
        public class MyResetAgentTask : MyTask<MyGridWorld>
        {
            private int m_positionX = 0;
            [MyBrowsable, Category("Position")]
            [YAXSerializableField(DefaultValue = 0)]
            public int PositionX
            {
                get
                {
                    return m_positionX;
                }
                set
                {
                    if (Owner != null && Owner.World != null && value >= 0 && value < Owner.World.GetWidth())
                    {
                        m_positionX = value;
                    }
                }
            }

            private int m_positionY = 0;
            [MyBrowsable, Category("Position")]
            [YAXSerializableField(DefaultValue = 0)]
            public int PositionY
            {
                get
                {
                    return m_positionY;
                }
                set
                {
                    if (Owner != null && Owner.World != null && value >= 0 && value < Owner.World.GetHeight())
                    {
                        m_positionY = value;
                    }
                }
            }

            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                MyLog.INFO.WriteLine("Agent reset to position: [" + PositionX + "," + PositionY + "].");
                Owner.World.GetAgent().setPosition(new int2(PositionX, PositionY));
            }
        }
    }
}
