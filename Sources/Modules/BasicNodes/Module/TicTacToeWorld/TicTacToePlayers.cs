using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;

namespace GoodAI.Modules.TicTacToe
{
    class MyTicTacToeGame
    {
        public static MyTicTacToeWorld.PLAYERS COMPUTER = MyTicTacToeWorld.PLAYERS.PLAYER_O;
        public static MyTicTacToeWorld.PLAYERS PLAYER = MyTicTacToeWorld.PLAYERS.PLAYER_X;

        public static readonly int[,] allLines = new int[,]{
                {0,1,2},
                {3,4,5},
                {6,7,8},
                {0,4,8},
                {6,4,2},
                {0,3,6},
                {1,4,7},
                {2,5,8} };

        public MyTicTacToeGame(int[] currentState)
        {
            if (currentState.Length != MyTicTacToeWorld.NO_POSITIONS)
            {
                MyLog.ERROR.WriteLine("Incorrect length of current state, expected 9 positions");
            }
        }

        public bool ApplyAction(int where, MyTicTacToeWorld.PLAYERS who, int[] currentState)
        {
            if (currentState[where] != 0)
            {
                return false;
            }
            currentState[where] = (int)who;
            return true;
        }

        public bool CheckWinner(MyTicTacToeWorld.PLAYERS who, int[] currentState)
        {
            for (int i = 0; i < allLines.GetLength(0); i++)
            {
                for (int j = 0; j < allLines.GetLength(1); j++)
                {
                    if (currentState[allLines[i, j]] != (int)who)
                    {
                        break;
                    }
                    if (j == allLines.GetLength(1) - 1)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        public static bool NoFreePlace(int[] currentState)
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

        public bool Ended(int[] currentState)
        {
            if (NoFreePlace(currentState))
                return true;
            if (CheckWinner(PLAYER, currentState))
                return true;
            if (CheckWinner(COMPUTER, currentState))
                return true;
            return false;
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
    /// Thing, which plays the tic tac toe
    /// </summary>
    interface ITicTacToeEngine
    {
        int GenerateAction(int[] currentState);
    }

    interface IConfigurableTicTacToeEngine : ITicTacToeEngine
    {
        void UpdateDifficulty(float difficulty);
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

    class MyEngineA : IConfigurableTicTacToeEngine
    {
        public static readonly int MAX_DEPTH = 4;

        private float m_difficulty;
        private MyTicTacToeGame m_game;

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
            if (moves.Count >= currentState.Length - 1)
                return r.Next(moves.Count);

            // action is random with P = 1-difficulty
            if (r.NextDouble() > m_difficulty)
                return moves[r.Next(moves.Count)];

            // I can win now
            int win = CanWinByMoving(currentState, MyTicTacToeGame.COMPUTER);
            if (win != -1)
                return win;

            // have to defend now
            win = CanWinByMoving(currentState, MyTicTacToeGame.PLAYER);
            if (win != -1)
                return win;

            float maxVal = float.MinValue;
            float val;

            int[] tmpState = new int[currentState.Length];

            List<int> bestIndexes = new List<int>();
            bestIndexes.Add(0);

            // run minmax for all possible moves, get (one of the) max value(s)
            for (int i = 0; i < moves.Count; i++)
            {
                currentState.CopyTo(tmpState, 0);
                tmpState[moves[i]] = (int)MyTicTacToeGame.COMPUTER;

                //val = EvaluateState(tmpState, MyTicTacToeGame.COMPUTER);
                val = minimax(tmpState, 1, MAX_DEPTH, false);

                if (val == maxVal)
                {
                    bestIndexes.Add(i);
                }
                else if (val > maxVal)
                {
                    maxVal = val;
                    bestIndexes.Clear();
                    bestIndexes.Add(i);
                }
            }

            // if multiple identically good moves found, choose randomly
            return moves[bestIndexes[r.Next(bestIndexes.Count)]];
        }

        /// <summary>
        /// If a given player can immediately win by moving an a particular place
        /// </summary>
        /// <param name="currentState">state of the board</param>
        /// <param name="who">player who can win</param>
        /// <returns>-1 if cannot win in one move, corrspoding board position otherwise</returns>
        private int CanWinByMoving(int[] currentState, MyTicTacToeWorld.PLAYERS who)
        {
            int myPlaces, emptyPlace, emptyPos = 0;

            for (int i = 0; i < MyTicTacToeGame.allLines.GetLength(0); i++)
            {
                myPlaces = 0;
                emptyPlace = 0;
                for (int j = 0; j < MyTicTacToeGame.allLines.GetLength(1); j++)
                {
                    if (currentState[MyTicTacToeGame.allLines[i, j]] == 0)
                    {
                        emptyPlace++;
                        emptyPos = MyTicTacToeGame.allLines[i, j];
                    }
                    else if (currentState[MyTicTacToeGame.allLines[i, j]] == (int)who)
                    {
                        myPlaces++;
                    }
                    if (myPlaces == 2 && emptyPlace == 1)
                    {
                        return emptyPos;
                    }
                }
            }
            return -1;
        }

        /// <summary>
        /// returns the value of a given game state for given player
        /// </summary>
        /// <param name="state">game state to be evaluated</param>
        /// <param name="indexes">list of 3 indexes (row, column, diagonal)</param>
        /// <param name="player">COMPUTER or PLAYERS</param>
        /// <returns>value of the state</returns>
        private float EvalSet(int[] state, int[] indexes, MyTicTacToeWorld.PLAYERS player)
        {
            if (indexes.Length != 3)
            {
                MyLog.ERROR.WriteLine("length of indexes has to be 3!");
                return 0;
            }
            int freeOnes = 0;
            int myOnes = 0;
            int oponentOnes = 0;

            for (int i = 0; i < indexes.Length; i++)
            {
                if (state[indexes[i]] == (int)player)
                {
                    myOnes++;
                }
                else if (state[indexes[i]] == 0)
                {
                    freeOnes++;
                }
                else
                {
                    oponentOnes++;
                }
            }
            if (myOnes == 1 && freeOnes == 2)       // good
                return 1;
            else if (myOnes == 2 && freeOnes == 1)  // better
                return 10;
            else if (myOnes == 3)                   // win
                return 100;
            return 0;
        }


        private float EvaluateState(int[] state, MyTicTacToeWorld.PLAYERS player)
        {
            float sum = 0;
            int[] indexes = new int[3];

            for (int i = 0; i < MyTicTacToeGame.allLines.GetLength(0); i++)
            {
                for (int j = 0; j < indexes.Length; j++)
                    indexes[j] = MyTicTacToeGame.allLines[i, j];

                sum += EvalSet(state, indexes, player);
            }
            return sum;
        }

        private float minimax(int[] state, int depth, int maxDepth, bool computerPlays)
        {
            // terminating condition?
            if (depth == maxDepth || m_game.Ended(state))
            {
                if (computerPlays)
                    return -this.EvaluateState(state, MyTicTacToeGame.PLAYER);

                return this.EvaluateState(state, MyTicTacToeGame.COMPUTER);
            }

            // recursion
            List<int> moves = m_game.GetPossibleMoves(state);
            int[] tmpState = new int[state.Length];
            float val;
            float tmp;

            if (computerPlays)
            {
                val = int.MinValue;

                for (int i = 0; i < moves.Count; i++)
                {
                    state.CopyTo(tmpState, 0);
                    tmpState[moves[i]] = (int)MyTicTacToeGame.COMPUTER;

                    tmp = minimax(tmpState, depth + 1, maxDepth, false);
                    if (tmp > val)
                        val = tmp;
                }
                return val;
            }
            else
            {
                val = int.MaxValue;

                for (int i = 0; i < moves.Count; i++)
                {
                    state.CopyTo(tmpState, 0);
                    tmpState[moves[i]] = (int)MyTicTacToeGame.PLAYER;

                    tmp = minimax(tmpState, depth + 1, maxDepth, true);
                    if (tmp < val)
                        val = tmp;
                }
                return val;
            }
        }
    }
}
