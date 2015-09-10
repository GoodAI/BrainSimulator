using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.TicTacToe
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>Node that plays the Tic Tac Toe game in MyTicTacToeWorld.</summary>
    /// <description>
    /// 
    /// Connect to the state output and corresponding input, this node will play the TicTacToe.
    /// Choose whether it plays with X or O before the simulation (has to correpsond with the World input).
    /// 
    /// The node uses combination of simple rules with heuristic-based minimax algorithm searching 4 moves ahead.
    /// 
    /// The node should be used in the ConditionalGroup, which is triggered by the PlayerX/O signal.
    /// 
    /// <h3>Outputs:</h3>
    /// <ul>
    ///     <li> <b>ActionOutput:</b>OneOfN code specifying the position where to place the X/Y.</li>
    /// </ul>
    /// 
    /// <h3>Inputs</h3>
    /// <ul>
    ///     <li> <b>StateInput:</b>9 numbers that have values: {0 (empty), 1 (O), 2 (X)}</li>
    /// </ul>
    /// 
    /// </description>
    class MyTicTacToePlayerNode : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> StateInput
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> ActionOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        MyTicTacToeWorld.PLAYERS me;

        [MyBrowsable, Category("I Play With")]
        [YAXSerializableField(DefaultValue = MyTicTacToeWorld.PLAYERS.PLAYER_X)]
        public MyTicTacToeWorld.PLAYERS Me
        {
            get
            {
                return me;
            }
            set
            {
                me = value;
            }
        }

        protected int[] m_currentState;
        protected int m_selectedAction;
        protected MyTicTacToeGame game;
        protected Random r;

        public override void UpdateMemoryBlocks()
        {
            ActionOutput.Count = MyTicTacToeWorld.NO_POSITIONS;
            m_currentState = new int[MyTicTacToeWorld.NO_POSITIONS];
            game = new MyTicTacToeGame(m_currentState);
            r = new Random();
        }

        public override void Validate(MyValidator validator)
        {
            if (StateInput == null)
            {
                validator.AddError(this, "StateInput is not connected");
            }
            else if (StateInput.Count != MyTicTacToeWorld.NO_POSITIONS)
            {
                validator.AddError(this, "Incorrect length of StateInput, expected 9, found " + StateInput.Count);
            }
        }

        private void DecodeState()
        {
            StateInput.SafeCopyToHost();

            for (int i = 0; i < StateInput.Host.Length; i++)
            {
                if ((int)StateInput.Host[i] != (int)MyTicTacToeWorld.TALES.EMPTY &&
                    (int)StateInput.Host[i] != (int)MyTicTacToeWorld.TALES.PLAYER_O &&
                    (int)StateInput.Host[i] != (int)MyTicTacToeWorld.TALES.PLAYER_X)
                {
                    MyLog.DEBUG.WriteLine("Unexpected state value, expected only these: " +
                        (int)MyTicTacToeWorld.TALES.EMPTY + ", " +
                        (int)MyTicTacToeWorld.PLAYERS.PLAYER_X + ", " +
                        (int)MyTicTacToeWorld.PLAYERS.PLAYER_O);

                    m_currentState[i] = (int)MyTicTacToeWorld.TALES.EMPTY;
                }
                else
                {
                    m_currentState[i] = (int)StateInput.Host[i];
                }
            }
        }

        private void EncodeAction()
        {
            ActionOutput.SafeCopyToDevice();
            Array.Copy(new int[MyTicTacToeWorld.NO_POSITIONS], ActionOutput.Host, MyTicTacToeWorld.NO_POSITIONS);
            ActionOutput.Host[m_selectedAction] = (int)Me;
            ActionOutput.SafeCopyToDevice();
        }


        public MyPlayerTask DetectDifferences { get; private set; }

        /// <summary>
        /// TicTacToe player.
        /// 
        /// <h3>Parameters</h3>
        /// <ul>
        ///     <li> <b>StateInput:</b>How well the computer plays. 1=good, 0=completely random.</li>
        /// </ul>
        /// </summary>
        [MyTaskInfo(OneShot = false)]
        public class MyPlayerTask : MyTask<MyTicTacToePlayerNode>
        {
            [MyBrowsable, Category("Configuration"),
                      Description("How well the computer plays. 1=good, 0=completely random")]
            [YAXSerializableField(DefaultValue = 0.5f)]
            public float Difficulty { get; set; }

            private ITicTacToeEngine m_computer;

            public override void Init(int nGPU)
            {
                m_computer = new MyEngineA(Difficulty, Owner.game);
            }

            private void UpdateDifficulty()
            {
                if (m_computer is IConfigurableTicTacToeEngine)
                    ((IConfigurableTicTacToeEngine)m_computer).UpdateDifficulty(Difficulty);
            }

            public override void Execute()
            {
                Owner.DecodeState();
                UpdateDifficulty();

                if (MyTicTacToeGame.NoFreePlace(Owner.m_currentState))
                {
                    Owner.m_selectedAction = 0;
                    Owner.EncodeAction();
                    return;
                }

                Owner.m_selectedAction = m_computer.GenerateAction(Owner.m_currentState);

                while (!Owner.game.ApplyAction(Owner.m_selectedAction, Owner.Me, Owner.m_currentState))
                {
                    Owner.m_selectedAction = m_computer.GenerateAction(Owner.m_currentState);
                }

                Owner.EncodeAction();
            }
        }
    }
}
