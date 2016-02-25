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
    /// <meta>mp</meta>
    /// <status>Working</status>
    /// <summary>Custom implementation of pong for two players.</summary>
    /// <description>Modifies CustomPongWorld by adding a second player. There are no bricks, just two paddles and a ball.
    /// Inspired by the <a href="https://en.wikipedia.org/wiki/Pong">original pong</a>.
    /// 
    /// <h3>Controls - Player A</h3>
    /// <ol>
    /// <li>Go left</li>
    /// <li>Stop</li>
    /// <li>Go right</li>
    /// </ol>
    /// 
    /// <h3>ControlsB - Player B</h3>
    /// <ol>
    /// <li>Go left</li>
    /// <li>Stop</li>
    /// <li>Go right</li>
    /// </ol>
    /// 
    /// BinaryEvent output has a vector of binary events as follows: "bounce ball", "opponent lost life", "lost life".<br />
    /// 
    /// Event output has values defined by UpdateTwoPlayerTask's parameters.<br />
    /// 
    /// PaddlePos, Event and BinaryEvent outputs are present twice, once for each of the players (A,B)<br />
    /// </description>
    public class MyTwoPlayerPongWorld : MyCustomPongWorld, IMyCustomTaskFactory
    {
        // the override (of Controls) is needed to avoid errors during validation 
        // that otherwise occur when an input is not connected to Controls or ControlsB
        [MyInputBlock]
        public override MyMemoryBlock<float> Controls 
        {
            get { return GetInput(0); }
        }

        [MyInputBlock]
        public MyMemoryBlock<float> ControlsB
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(7)]
        public MyMemoryBlock<float> PaddleBPosX
        {
            get { return GetOutput(7); }
            set { SetOutput(7, value); }
        }

        [MyOutputBlock(8)]
        public MyMemoryBlock<float> PaddleBPosY
        {
            get { return GetOutput(8); }
            set { SetOutput(8, value); }
        }

        [MyOutputBlock(9)]
        public MyMemoryBlock<float> EventB
        {
            get { return GetOutput(9); }
            set { SetOutput(9, value); }
        }

        [MyOutputBlock(10)]
        public MyMemoryBlock<float> BinaryEventB
        {
            get { return GetOutput(10); }
            set { SetOutput(10, value); }
        }

        [MyBrowsable, Category("Params"), ReadOnly(true)]
        [YAXSerializableField(DefaultValue = false)]
        public override bool BricksEnabled { get { return false; } set { } }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            PaddleBPosX.Count = 1;
            PaddleBPosY.Count = 1;
            EventB.Count = 1;
            BinaryEvent.Count = 3; // bounce ball, opponent lost life, lost life
            BinaryEventB.Count = 3;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (ControlsB != null)
            {
                validator.AssertError(ControlsB.Count >= 3, this, "Not enough controls (3 expected)");
            }
        }

        public virtual void CreateTasks() // overrides default behaviour, which would create new InitGameTask() and new UpdateTask().
        {
            InitGameTask = new MyInitTwoPlayerTask();
            UpdateTask = new MyUpdateTwoPlayerTask();
        }

        /// <summary>
        /// Loads textures, prepares game objects.
        /// </summary>
        [MyTaskInfo(OneShot = true)]
        public class MyInitTwoPlayerTask : MyInitTask
        {

            public override void Init(int nGPU)
            {
                base.Init(nGPU);
            }

            public override void InitGameObjects()
            {
                base.InitGameObjects();
                AddPaddleBToGameObjects();
            }

            private void AddPaddleBToGameObjects()
            {
                MyTwoPlayerPongWorld Owner = (MyTwoPlayerPongWorld)this.Owner;

                if (Owner.m_gameObjects.Count < 3)
                {
                    MyGameObject paddleA = Owner.m_gameObjects[1];
                    MyGameObject paddleB = new MyGameObject()
                    {
                        pixelSize = new int2(paddleA.pixelSize.x, paddleA.pixelSize.y),
                        bitmap = paddleA.bitmap
                    };
                    Owner.m_gameObjects.Add(paddleB);
                }
            }

            public override void Execute()
            {
                base.Execute();
            }
        }

        /// <summary>
        /// Updates the game's state based on inputs. It allows configuration of Event output through properties.
        /// </summary>
        public class MyUpdateTwoPlayerTask : MyUpdateTask
        {
            [MyBrowsable, Category("Events")]
            [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
            public float OPPONENT_LOST_LIFE { get; set; }

            [MyBrowsable(false)] // hide BRICK_DESTROYED property from the propertyGrid
            public override float BRICK_DESTROYED { get; set; }

            MyTwoPlayerPongWorld GetTwoPlayerOwner() 
            { 
                return (MyTwoPlayerPongWorld)Owner; 
            }

            protected static readonly int OPPONENT_LOST_LIFE_I = BRICK_DESTROYED_I; // reuse brick_destroyed memory

            protected static readonly int PLAYER_A = 0;
            protected static readonly int PLAYER_B = 1;

            private int stepsFrozen = 0;

            protected int m_lastLost; // PLAYER_A / PLAYER_B
            protected int m_playerScore;
            protected int m_playerBScore;

            protected float m_controlBCoolDown = 0;
            protected float m_controlB;

            public override void Init(int nGPU)
            {
                base.Init(nGPU);
                m_lastLost = -1;
                m_playerBScore = 0;
                m_playerScore = 0;
                m_controlB = 0;
                m_controlBCoolDown = 0;
            }

            protected override void ExecutePrepareHost()
            {
                base.ExecutePrepareHost();

                MyTwoPlayerPongWorld Owner = GetTwoPlayerOwner();

                Owner.BinaryEventB.Host[LOST_LIFE_I] = 0;
                Owner.BinaryEventB.Host[BOUNCE_BALL_I] = 0;
                Owner.BinaryEventB.Host[OPPONENT_LOST_LIFE_I] = 0;
                Owner.EventB.Host[0] = 0.0f;

                Owner.ControlsB.SafeCopyToHost();
            }

            protected override void ExecuteResolveEvents()
            {
                MyTwoPlayerPongWorld Owner = GetTwoPlayerOwner();

                MyGameObject ball = Owner.m_gameObjects[0];
                MyGameObject paddle = Owner.m_gameObjects[1];
                MyGameObject paddleB = Owner.m_gameObjects[2];

                ResolveBallEvents(ball, paddle, paddleB);

                UpdateControl(ref m_control, ref m_controlCoolDown, Owner.Controls.Host);
                UpdateControl(ref m_controlB, ref m_controlBCoolDown, Owner.ControlsB.Host);
                ResolvePaddleEvents(paddle, m_control);
                ResolvePaddleEvents(paddleB, m_controlB);

                Owner.BallPosX.Host[0] = ball.position.x + ball.pixelSize.x * 0.5f;
                Owner.BallPosY.Host[0] = ball.position.y + ball.pixelSize.y * 0.5f;

                Owner.PaddlePosX.Host[0] = paddle.position.x + paddle.pixelSize.x * 0.5f;
                Owner.PaddlePosY.Host[0] = paddle.position.y + paddle.pixelSize.y * 0.5f;

                Owner.PaddleBPosX.Host[0] = paddleB.position.x + paddleB.pixelSize.x * 0.5f;
                Owner.PaddleBPosY.Host[0] = paddleB.position.y + paddleB.pixelSize.y * 0.5f;
            }

            protected override void ExecuteCopyToDevice()
            {
                base.ExecuteCopyToDevice();

                MyTwoPlayerPongWorld Owner = GetTwoPlayerOwner();

                Owner.PaddleBPosX.SafeCopyToDevice();
                Owner.PaddleBPosY.SafeCopyToDevice();

                Owner.EventB.SafeCopyToDevice();
                Owner.BinaryEventB.SafeCopyToDevice();
            }

            public override void Execute()
            {
                base.Execute();
            }

            public override void ResetGame()
            {
                ResetBallAndPaddle();

                if(m_lastLost == PLAYER_A)
                    m_playerBScore++;
                else if (m_lastLost == PLAYER_B)
                    m_playerScore++;
            }

            protected override void ResetBallAndPaddle()
            {
                base.ResetBallAndPaddle(); // moves the ball on the paddle of player A

                MyTwoPlayerPongWorld Owner = GetTwoPlayerOwner();

                MyGameObject ball = Owner.m_gameObjects[0]; 
                MyGameObject paddleB = Owner.m_gameObjects[2];

                if (m_lastLost == PLAYER_A) // moves the ball on the paddle of player B
                {
                    ball.position.y = 22 - ball.pixelSize.y;
                    ball.velocity.y = 1f;
                }

                paddleB.position.x = (Owner.Scene.Width - paddleB.pixelSize.x) * 0.5f;
                paddleB.position.y = 14 - paddleB.pixelSize.y;

                paddleB.velocity.x = 0;
                paddleB.velocity.y = 0;

                m_controlB = m_controlBCoolDown = 0;
            }

            protected void ResolveBallEvents(MyGameObject ball, MyGameObject paddle, MyGameObject paddleB)
            {
                MyTwoPlayerPongWorld Owner = GetTwoPlayerOwner();

                float2 futurePos = ball.position + ball.velocity;

                //leftSide
                if (futurePos.x < 0 && ball.velocity.x < 0)
                {
                    ball.velocity.x = -ball.velocity.x;
                }
                //rightSide
                if (futurePos.x + ball.pixelSize.x > Owner.Scene.Width && ball.velocity.x > 0)
                {
                    ball.velocity.x = -ball.velocity.x;
                }

                //bottom side
                if (futurePos.y + ball.pixelSize.y > Owner.Scene.Height && ball.velocity.y > 0)
                {
                    if (stepsFrozen == 0)
                    {
                        Owner.Event.Host[0] += LOST_LIFE; // take the life at the first freeze frame
                        Owner.EventB.Host[0] += OPPONENT_LOST_LIFE; // reward for the other player
                        Owner.BinaryEvent.Host[LOST_LIFE_I] = 1;
                        Owner.BinaryEventB.Host[OPPONENT_LOST_LIFE_I] = 1;
                        m_lastLost = PLAYER_A;
                    }
                    if (stepsFrozen == this.FreezeAfterFail)
                    {
                        stepsFrozen = 0;
                        ResetGame();
                    }
                    else
                    {
                        stepsFrozen++;
                        return;
                    }
                }

                //top side
                if (futurePos.y < 0 && ball.velocity.y < 0)
                {
                    if (stepsFrozen == 0)
                    {
                        Owner.EventB.Host[0] += LOST_LIFE; // take the life at the first freeze frame
                        Owner.Event.Host[0] += OPPONENT_LOST_LIFE; // reward for the other player
                        Owner.BinaryEventB.Host[LOST_LIFE_I] = 1;
                        Owner.BinaryEvent.Host[OPPONENT_LOST_LIFE_I] = 1;
                        m_lastLost = PLAYER_B;
                    }
                    if (stepsFrozen == this.FreezeAfterFail)
                    {
                        stepsFrozen = 0;
                        ResetGame();
                    }
                    else
                    {
                        stepsFrozen++;
                        return;
                    }
                }

                //paddle A
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

                // paddle B
                if (futurePos.y < paddleB.position.y + paddleB.pixelSize.y &&
                    futurePos.y > paddleB.position.y &&
                    futurePos.x + 10 > paddleB.position.x &&
                    futurePos.x + ball.pixelSize.x < paddleB.position.x + paddleB.pixelSize.x + 10 &&
                    ball.velocity.y < 0)
                {
                    ball.velocity.y = -ball.velocity.y;
                    ball.velocity.x += paddleB.velocity.x * 0.2f;

                    Owner.EventB.Host[0] += BOUNCE_BALL;
                    Owner.BinaryEventB.Host[BOUNCE_BALL_I] = 1;
                }

                ball.position += ball.velocity * DELTA_T;
            }
        }

    }
}
