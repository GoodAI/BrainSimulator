using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CustomModels.TicTacToeWorld
{
    class MyTicTacToeGame
    {
        private readonly int NO_POSITIONS = 9;

        public static readonly int EMPTY = 0, TALEO = 1, TALEX = 2;
        public static int COMPUTER = TALEO, PLAYER = TALEX;

        private int[,] winners = new int[,]{
                {0,1,2},
                {3,4,5},
                {6,7,8},
                {0,4,8},
                {6,4,2},
                {0,3,6},
                {2,5,8} };

        public MyTicTacToeGame(int[] currentState)
        {
            if (currentState.Length != NO_POSITIONS)
            {
                MyLog.ERROR.WriteLine("Incorrect length of current state, expected 9 positions");
            }
        }

        public bool ApplyAction(int where, int who, int[] currentState)
        {
            if (currentState[where] != 0)
            {
                return false;
            }
            currentState[where] = who;
            return true;
        }

        public bool CheckWinner(int who, int[] currentState)
        {
            for (int i = 0; i < winners.GetLength(0); i++)
            {
                for (int j = 0; j < winners.GetLength(1); j++)
                {
                    if (currentState[winners[i, j]] != who)
                    {
                        break;
                    }
                    if (j == winners.GetLength(1) - 1)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        public bool Ended(int[] currentState)
        {
            for (int i = 0; i < currentState.Length; i++)
            {
                if (currentState[i] == 0)
                {
                    return false;
                }
            }
            return true;
        }

        public List<int> GetPossibleMoves(int[] state)
        {
            List<int> moves = new List<int>();
            for (int i = 0; i < state.Length; i++)
            {
                if (state[i] == 0)
                {
                    moves.Add(i);
                }
            }
            return moves;
        }

    }

    /// <summary>
    /// Thing, that plays the tic tac toe
    /// </summary>
    interface ITicTacToeEngine
    {
        int GenerateAction(int[] currentState);
    }

    class MyEngineRandom : ITicTacToeEngine
    {
        private Random r = new Random();

        public int GenerateAction(int[] currentState)
        {
            List<int> freePlaces = new List<int>();
            for (int i = 0; i < currentState.Length; i++)
            {
                if (currentState[i] == 0)
                {
                    freePlaces.Add(i);
                }
            }

            int ac = r.Next(freePlaces.Count);
            return freePlaces[ac];
        }
    }

    class MyEngineA : ITicTacToeEngine
    {
        private float m_difficulty;
        private MyTicTacToeGame m_game;

        public static float VAL = 100.0f;

        private Random r = new Random();

        public MyEngineA(float difficulty, MyTicTacToeGame game)
        {
            this.m_game = game;
            this.UpdateDifficulty(difficulty);
        }

        public void UpdateDifficulty(float difficulty)
        {
            if (difficulty < 0.0 || difficulty > 1.0)
            {
                MyLog.DEBUG.WriteLine("Warning, difficulty parameter should be in <0,1>");

                if (difficulty < 0.0)
                    this.m_difficulty = 0.0f;
                else if (difficulty > 1.0)
                    this.m_difficulty = 1.0f;
            }
            this.m_difficulty = difficulty;
        }

        public int GenerateAction(int[] currentState)
        {
            List<int> moves = m_game.GetPossibleMoves(currentState);

            if (moves.Count == 0)
            {
                MyLog.ERROR.WriteLine("No move is possible!");
                return 0;
            }

            // first move generated randomly
            if (moves.Count == currentState.Length)
            {
                return r.Next(currentState.Length);
            }

            // action is random with P = 1-difficulty
            if (r.NextDouble() > m_difficulty)
            {
                return moves[r.Next(moves.Count)];
            }


            int bestMove = 0;
            float maxVal = float.MinValue;
            float val;

            int[] tmpState = new int[currentState.Length];

            for (int i = 0; i < moves.Count; i++)
            {
                currentState.CopyTo(tmpState,0);
                tmpState[moves[i]] = MyTicTacToeGame.COMPUTER;

                val = minimax(tmpState, 1, true);
                MyLog.DEBUG.WriteLine("value of move "+moves[i]+" is "+val);
                
                if (val > maxVal)
                {
                    maxVal = val;
                    bestMove = i;
                }
            }

            MyLog.DEBUG.WriteLine("best action is " + moves[bestMove] + " and value " + maxVal);
            return moves[bestMove];
        }

        private float minimax(int[] state, int depth, bool iPlay)
        {
            // termination conditions
            if(m_game.CheckWinner(MyTicTacToeGame.COMPUTER, state))
            {
                return VAL/depth;
            }
            else if(m_game.CheckWinner(MyTicTacToeGame.PLAYER, state))
            {
                return -VAL/depth;
            }
            else if(m_game.Ended(state))
            {
                return 0;
            }

            // recursion
            List<int> moves = m_game.GetPossibleMoves(state);
            int[] tmpState = new int[state.Length];
            float val;
            float tmp;

            if (iPlay)
            {
                val = int.MinValue;

                for (int i = 0; i < moves.Count; i++)
                {
                    state.CopyTo(tmpState, 0);
                    tmpState[moves[i]] = MyTicTacToeGame.COMPUTER;
                    
                    tmp = minimax(tmpState, depth + 1, false);
                    if (tmp > val)
                        val = tmp;
                }
            }
            else
            {
                val = int.MaxValue;

                for (int i = 0; i < moves.Count; i++)
                {
                    state.CopyTo(tmpState, 0);
                    tmpState[moves[i]] = MyTicTacToeGame.PLAYER;
                    
                    tmp = minimax(tmpState, depth + 1, true);
                    if (tmp < val)
                        val = tmp;
                }
            }
            return val;
        }
    }

}
