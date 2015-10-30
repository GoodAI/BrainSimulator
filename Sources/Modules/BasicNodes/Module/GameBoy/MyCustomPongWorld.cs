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
    /// <status>Working</status>
    /// <summary>Custom implementation of the pong (arkanoid) game.</summary>
    /// <description>Follows the original game boy EVA pong game (same graphics &amp; levels). Bricks can be turned on and off. Ball can be fired in arbitrary direction. It is faster.</description>
    public class MyCustomPongWorld : MyWorld
    {
        public class MyGameObject
        {
            public float2 position;
            public int2 pixelSize;
            public float2 velocity;
            public CUdeviceptr bitmap;
        };

        [MyInputBlock]
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
        public MyMemoryBlock<float> BallPosX
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> BallPosY
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> PaddlePosX
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }

        [MyOutputBlock(5)]
        public MyMemoryBlock<float> PaddlePosY
        {
            get { return GetOutput(5); }
            set { SetOutput(5, value); }
        }

        [MyOutputBlock(6)]
        public MyMemoryBlock<float> BinaryEvent
        {
            get { return GetOutput(6); }
            set { SetOutput(6, value); }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 160)]
        public int DISPLAY_WIDTH { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 140)]
        public int DISPLAY_HEIGHT { get; set; }

        public MyMemoryBlock<float> Bitmaps { get; private set; }

        private Dictionary<string, Bitmap> m_bitmapTable = new Dictionary<string, Bitmap>();
        private string m_errorMessage;

        private List<MyGameObject> m_gameObjects;

        public MyMemoryBlock<int> Bricks { get; private set; }
        private MyGameObject m_brickPrototype;
        private MyGameObject m_lifePrototype;

        private const int BRICKS_COUNT_X = 10;
        private const int BRICKS_COUNT_Y = 10;

        private int m_level = 0;
        private int m_lifes = 0;
        private int m_bricksRemains = 0;

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = false)]
        public bool BricksEnabled { get; set; }

        public override void UpdateMemoryBlocks()
        {
            Visual.Count = DISPLAY_WIDTH * DISPLAY_HEIGHT;
            Visual.ColumnHint = DISPLAY_WIDTH;

            Bitmaps.Count = 0;

            Bitmaps.Count += LoadAndGetBitmapSize(@"res\pong\ball.png");
            Bitmaps.Count += LoadAndGetBitmapSize(@"res\pong\paddle.png");
            Bitmaps.Count += LoadAndGetBitmapSize(@"res\pong\brick.png");
            Bitmaps.Count += LoadAndGetBitmapSize(@"res\pong\life.png");

            BallPosX.Count = 1;
            BallPosY.Count = 1;
            PaddlePosX.Count = 1;
            PaddlePosY.Count = 1;
            Event.Count = 1;
            BinaryEvent.Count = 3;

            Bricks.Count = BRICKS_COUNT_X * BRICKS_COUNT_Y;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            if (Controls != null)
            {
                validator.AssertError(Controls.Count >= 3, this, "Not enough controls");
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
        /// Initialises the game state and loads the resources.
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MyCustomPongWorld>
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

                Bitmap bitmap = Owner.m_bitmapTable[@"res\pong\ball.png"];

                MyGameObject ball = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[@"res\pong\paddle.png"];

                MyGameObject paddle = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[@"res\pong\brick.png"];

                Owner.m_brickPrototype = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                    position = new float2(0, 16)
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                bitmap = Owner.m_bitmapTable[@"res\pong\life.png"];

                Owner.m_lifePrototype = new MyGameObject()
                {
                    pixelSize = new int2(bitmap.Width, bitmap.Height),
                    bitmap = devBitmaps.DevicePointer + devBitmaps.TypeSize * offset,
                    position = new float2(0, 3)
                };
                offset += FillWithChannelFromBitmap(bitmap, 0, Owner.Bitmaps.Host, offset);

                Owner.m_gameObjects.Add(ball);
                Owner.m_gameObjects.Add(paddle);

                Owner.Bitmaps.SafeCopyToDevice();

                Owner.m_lifes = 0;
                Owner.m_level = 0;
                Owner.UpdateTask.ResetGame();
            }
        }

        /// <summary>
        /// Renders the game to visual output.
        /// </summary>
        public class MyRenderTask : MyTask<MyCustomPongWorld>
        {
            private MyCudaKernel m_kernel;
            private MyCudaKernel m_bricksKernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "DrawSpriteKernel");
                m_bricksKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\CustomPong", "DrawBricksKernel");
            }

            public override void Execute()
            {
                Owner.Visual.Fill(1.0f);

                if (Owner.BricksEnabled)
                {
                    MyGameObject brick = Owner.m_brickPrototype;

                    m_bricksKernel.SetupExecution(brick.pixelSize.x * brick.pixelSize.y);
                    m_bricksKernel.Run(Owner.Visual, Owner.DISPLAY_WIDTH, Owner.DISPLAY_HEIGHT,
                        Owner.Bricks, BRICKS_COUNT_X, BRICKS_COUNT_Y,
                        brick.bitmap, brick.position, brick.pixelSize);
                }

                for (int i = 0; i < Owner.m_gameObjects.Count; i++)
                {
                    MyGameObject g = Owner.m_gameObjects[i];

                    m_kernel.SetupExecution(g.pixelSize.x * g.pixelSize.y);
                    m_kernel.Run(Owner.Visual, Owner.DISPLAY_WIDTH, Owner.DISPLAY_HEIGHT, g.bitmap, g.position, g.pixelSize);
                }

                MyGameObject life = Owner.m_lifePrototype;
                m_kernel.SetupExecution(life.pixelSize.x * life.pixelSize.y);

                /*
                for (int i = 0; i < Owner.m_lifes; i++)
                {
                    life.position.x = i * 8;
                    m_kernel.Run(Owner.Visual, Owner.DISPLAY_WIDTH, Owner.DISPLAY_HEIGHT, life.bitmap, life.position, life.pixelSize);
                }
                 * */
            }
        }

        /// <summary>
        /// BinaryEvent output has vector of binary events as follows: "bounce ball", "brick destroyed", "lost life".
        /// </summary>
        public class MyUpdateTask : MyTask<MyCustomPongWorld>
        {
            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = false)]
            public bool RandomBallDir { get; set; }

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0)]
            public int FreezeAfterFail { get; set; }

            private int stepsFrozen = 0;

            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 0.1f), YAXElementFor("Structure")]
            public float BOUNCE_BALL { get; set; }

            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
            public float LOST_LIFE { get; set; }

            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
            public float BRICK_DESTROYED { get; set; }

            private static readonly int BOUNCE_BALL_I = 0;
            private static readonly int BRICK_DESTROYED_I = 1;
            private static readonly int LOST_LIFE_I = 2;
            
            //probably unnecessary
            private const float MAX_PADDLE_VELOCITY = 4.0f;

            private const float INIT_BALL_VELOCITY = 1.3f;

            private static readonly float2 PADDLE_ACCELERATION = new float2(0.4f, 0);
            private const float PADDLE_FRICTION = 0.1f;
            private const float DELTA_T = 1.0f;

            private Random m_random = new Random();

            public override void Init(int nGPU)
            {

            }

            public override void Execute()
            {
                Owner.Event.Host[0] = 0;
                Owner.BinaryEvent.Host[BOUNCE_BALL_I] = 0;
                Owner.BinaryEvent.Host[LOST_LIFE_I] = 0;
                Owner.BinaryEvent.Host[BRICK_DESTROYED_I] = 0;

                MyGameObject ball = Owner.m_gameObjects[0];
                MyGameObject paddle = Owner.m_gameObjects[1];

                Owner.Controls.SafeCopyToHost();

                ResolveBallEvents(ball, paddle);

                if (Owner.BricksEnabled)
                {
                    ResolveBricksEvents(ball);
                }

                ResolvePaddleEvents(paddle, GetControl());

                Owner.BallPosX.Host[0] = ball.position.x + ball.pixelSize.x * 0.5f;
                Owner.BallPosY.Host[0] = ball.position.y + ball.pixelSize.y * 0.5f;

                Owner.PaddlePosX.Host[0] = paddle.position.x + paddle.pixelSize.x * 0.5f;
                Owner.PaddlePosY.Host[0] = paddle.position.y + paddle.pixelSize.y * 0.5f;

                Owner.BallPosX.SafeCopyToDevice();
                Owner.BallPosY.SafeCopyToDevice();
                Owner.PaddlePosX.SafeCopyToDevice();
                Owner.PaddlePosY.SafeCopyToDevice();

                Owner.Event.SafeCopyToDevice();
                Owner.BinaryEvent.SafeCopyToDevice();

                if (Owner.BricksEnabled)
                {
                    Owner.Bricks.SafeCopyToDevice();
                }
            }

            public void ResetGame()
            {
                ResetBallAndPaddle();

                Owner.m_lifes--;

                if (Owner.m_lifes < 0)
                {
                    Owner.m_lifes = 3;
                    Owner.m_level = 0;

                    SetLevel();
                }
            }

            private void ResetBallAndPaddle()
            {
                MyGameObject ball = Owner.m_gameObjects[0];
                MyGameObject paddle = Owner.m_gameObjects[1];

                ball.position.x = (Owner.DISPLAY_WIDTH - ball.pixelSize.x) * 0.5f;
                ball.position.y = Owner.DISPLAY_HEIGHT - 22;

                if (RandomBallDir)
                {
                    ball.velocity.x = (float)m_random.NextDouble() * 0.6f + 0.4f;
                    ball.velocity.x *= m_random.NextDouble() < 0.5f ? -1 : 1;
                }
                else
                {
                    ball.velocity.x = 1f;
                }
                ball.velocity.y = -1f;

                ball.velocity /= (float)Math.Sqrt(ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y);
                ball.velocity *= INIT_BALL_VELOCITY;

                paddle.position.x = (Owner.DISPLAY_WIDTH - paddle.pixelSize.x) * 0.5f;
                paddle.position.y = Owner.DISPLAY_HEIGHT - 14;

                paddle.velocity.x = 0;
                paddle.velocity.y = 0;
            }

            private void SetLevel()
            {
                if (Owner.BricksEnabled)
                {
                    Array.Copy(LEVELS[Owner.m_level], Owner.Bricks.Host, Owner.Bricks.Count);
                    Owner.Bricks.SafeCopyToDevice();

                    Owner.m_bricksRemains = LEVELS_BRICKS[Owner.m_level];
                }
            }

            private void ResolveBallEvents(MyGameObject ball, MyGameObject paddle)
            {
                float2 futurePos = ball.position + ball.velocity;

                //topSide
                if (futurePos.y < 0 && ball.velocity.y < 0)
                {
                    ball.velocity.y = -ball.velocity.y + (float)(m_random.NextDouble() * 0.2 - 0.1);
                }
                //leftSide
                if (futurePos.x < 0 && ball.velocity.x < 0)
                {
                    ball.velocity.x = -ball.velocity.x;
                }
                //rightSide
                if (futurePos.x + ball.pixelSize.x > Owner.DISPLAY_WIDTH && ball.velocity.x > 0)
                {
                    ball.velocity.x = -ball.velocity.x;
                }

                //bottom side
                if (futurePos.y + ball.pixelSize.y > Owner.DISPLAY_HEIGHT && ball.velocity.y > 0)
                {
                    if (stepsFrozen == 0)
                    {
                        Owner.Event.Host[0] += LOST_LIFE; // take the life at the first freeze frame
                        Owner.BinaryEvent.Host[LOST_LIFE_I] = 1;
                    }
                    if (stepsFrozen == this.FreezeAfterFail)
                    {
                        stepsFrozen = 0;
                        //Owner.Event.Host[0] += LOST_LIFE;
                        ResetGame();
                    }
                    else
                    {
                        stepsFrozen++;
                        return;
                    }
                }

                //paddle
                if (futurePos.y + ball.pixelSize.y > paddle.position.y &&
                    futurePos.y + ball.pixelSize.y < paddle.position.y + paddle.pixelSize.y &&
                    futurePos.x + 10 > paddle.position.x &&
                    futurePos.x + ball.pixelSize.x < paddle.position.x + paddle.pixelSize.x + 10 &&
                    ball.velocity.y > 0)
                {
                    ball.velocity.y = -ball.velocity.y;
                    ball.velocity.x += paddle.velocity.x * 0.2f;

                    Owner.Event.Host[0] += BOUNCE_BALL;
                    Owner.BinaryEvent.Host[BOUNCE_BALL_I] = 1;
                }

                ball.position += ball.velocity * DELTA_T;
            }

            private void ResolvePaddleEvents(MyGameObject paddle, float control)
            {
                paddle.velocity += (control * PADDLE_ACCELERATION - paddle.velocity * PADDLE_FRICTION) * DELTA_T;

                if (paddle.velocity.x > MAX_PADDLE_VELOCITY)
                {
                    paddle.velocity.x = MAX_PADDLE_VELOCITY;
                }
                else if (paddle.velocity.x < -MAX_PADDLE_VELOCITY)
                {
                    paddle.velocity.x = -MAX_PADDLE_VELOCITY;
                }

                float2 futurePos = paddle.position + paddle.velocity;

                if (futurePos.x < 0 || futurePos.x + paddle.pixelSize.x > Owner.DISPLAY_WIDTH)
                {
                    paddle.velocity.x = 0;
                }

                paddle.position += paddle.velocity * DELTA_T;
            }

            private void ResolveBricksEvents(MyGameObject ball)
            {
                float2 futurePos = ball.position + ball.velocity;

                if (!SolveBrickIntersection(futurePos.x, futurePos.y, ball))
                    if (!SolveBrickIntersection(futurePos.x + ball.pixelSize.x, futurePos.y, ball))
                        if (!SolveBrickIntersection(futurePos.x + ball.pixelSize.x, futurePos.y + ball.pixelSize.y, ball))
                            SolveBrickIntersection(futurePos.x, futurePos.y + ball.pixelSize.y, ball);

                if (Owner.m_bricksRemains <= 0)
                {
                    Owner.m_level = (Owner.m_level + 1) % LEVELS.Length;

                    SetLevel();
                    ResetBallAndPaddle();
                }
            }

            private bool SolveBrickIntersection(float contactX, float contactY, MyGameObject ball)
            {
                float relPosX = contactX - Owner.m_brickPrototype.position.x;
                float relPosY = contactY - Owner.m_brickPrototype.position.y;

                int bx = (int)(relPosX / Owner.m_brickPrototype.pixelSize.x);
                int by = (int)(relPosY / Owner.m_brickPrototype.pixelSize.y);
                int index = by * BRICKS_COUNT_X + bx;

                if (bx >= 0 && bx < BRICKS_COUNT_X && by >= 0 && by < BRICKS_COUNT_Y && Owner.Bricks.Host[index] > 0)
                {
                    Owner.Bricks.Host[index] = 0;
                    Owner.m_bricksRemains--;


                    // send brick destroyed event
                    Owner.Event.Host[0] += BRICK_DESTROYED;
                    Owner.BinaryEvent.Host[BRICK_DESTROYED_I] = 1;

                    float2 futureBallCenter = ball.position + ball.velocity;
                    futureBallCenter.x += ball.pixelSize.x * 0.5f;
                    futureBallCenter.y += ball.pixelSize.y * 0.5f;

                    float2 brickCenter = new float2(
                        (bx + 0.5f) * Owner.m_brickPrototype.pixelSize.x + Owner.m_brickPrototype.position.x,
                        (by + 0.5f) * Owner.m_brickPrototype.pixelSize.y + Owner.m_brickPrototype.position.y);

                    float2 contactDist = new float2(contactX - ball.velocity.x - brickCenter.x, contactY - ball.velocity.y - brickCenter.y);

                    float dotX = contactDist.x / Owner.m_brickPrototype.pixelSize.x;
                    float dotY = contactDist.y / Owner.m_brickPrototype.pixelSize.y;

                    if (Math.Abs(dotX) > Math.Abs(dotY))
                    {
                        ball.velocity.x = -ball.velocity.x;
                    }
                    else
                    {
                        ball.velocity.y = -ball.velocity.y;
                    }

                    return true;
                }
                else
                {
                    return false;
                }
            }

            private const float MAX_COOLDOWN = 5;
            private float m_controlCoolDown = 0;
            private float m_control;

            private float GetControl()
            {
                if (Owner.Controls.Host[0] > Owner.Controls.Host[1] && Owner.Controls.Host[0] > Owner.Controls.Host[2])
                {
                    m_control = -1;
                    m_controlCoolDown = MAX_COOLDOWN;
                }
                else if (Owner.Controls.Host[2] > Owner.Controls.Host[1])
                {
                    m_control = 1;
                    m_controlCoolDown = MAX_COOLDOWN;
                }
                else
                {
                    if (m_controlCoolDown < 0)
                    {
                        m_control = 0;
                    }
                    else
                    {
                        m_controlCoolDown--;
                    }
                }

                return m_control;
            }

            private static readonly int[] LEVEL_0 = 
            {                 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                               
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            };

            private static readonly int[] LEVEL_1 = 
            {                 
                1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
            };

            private static readonly int[] LEVEL_2 = 
            {                 
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            };

            private static readonly int[][] LEVELS = { LEVEL_0, LEVEL_1, LEVEL_2 };
            private static readonly int[] LEVELS_BRICKS = { 40, 40, 36 };
        }
    }
}
