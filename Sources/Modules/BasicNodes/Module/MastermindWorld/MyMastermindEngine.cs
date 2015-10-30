using ManagedCuda.BasicTypes;
using System;
using System.Diagnostics;
using System.Linq;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.MastermindWorld
{
    /// <summary>
    /// Holds parameters of the game world that are used both by the world and by the engine
    /// </summary>
    public sealed class MyWorldEngineParams
    {
        public int HiddenVectorLength;
        public int NumberOfColors;
        public int NumberOfGuesses;
        public int NumberOfRenderedGuesses;
        private string m_hiddenVectorUser;
        public string HiddenVectorUser
        {
            get { return m_hiddenVectorUser;  }
            set
            {
                if (value.Length > 0)
                {
                    // the potential format exception is handled higher in the call stack
                    HiddenVectorUserParsed = value.Trim().Split(',', ' ').Select(
                        a => int.Parse(a, CultureInfo.InvariantCulture)).ToArray();
                }
                else
                {
                    HiddenVectorUserParsed = null;
                }
                m_hiddenVectorUser = value;
            }
        }
        public int[] HiddenVectorUserParsed;
        public bool CanRepeatColors;
        public bool RepeatableHiddenVector;
    }

    /// <summary>
    /// Holds state and defines the behavior of all objects in the world.
    /// </summary>
    public sealed class MastermindWorldEngine
    {
        private MyWorldEngineParams m_params;
        private MyMastermindWorld m_world;

        private Random m_rndGen;

        public MastermindWorldEngine(MyMastermindWorld world, MyWorldEngineParams pars)
        {
            m_params = pars;
            m_world = world;
            if (pars.RepeatableHiddenVector)
                m_rndGen = new Random(100); // make it deterministic - always start with the same seed
            else
                m_rndGen = new Random();
        }

        /// <summary>
        /// called each time the game is reset
        /// </summary>
        private void ResetHiddenVector()
        {
            // (re)set value of the hidden vector:
            if (m_params.HiddenVectorUserParsed == null)
            {
                // generate a random hidden vector
                if (m_params.CanRepeatColors)
                {
                    for (int i = 0; i < m_params.HiddenVectorLength; i++)
                    {
                        m_world.HiddenVectorOutput.Host[i] = m_rndGen.Next(m_params.NumberOfColors);
                    }
                }
                else // Cannot repeat colors
                {
                    Debug.Assert(m_params.HiddenVectorLength <= m_params.NumberOfColors); // should be caught by validation

                    // colors from 0 to m_params.NumberOfColors-1
                    int[] availableColors = Enumerable.Range(0, m_params.NumberOfColors).ToArray(); 

                    // random permutation of colors
                    // http://stackoverflow.com/questions/1287567/is-using-random-and-orderby-a-good-shuffle-algorithm
                    int[] availableColorsShuffled = availableColors.OrderBy(x => m_rndGen.Next()).ToArray();

                    // copy just the first k elements of the permutation:
                    Array.Copy(availableColorsShuffled, m_world.HiddenVectorOutput.Host, m_params.HiddenVectorLength); 
                }
            }
            else
            {
                Array.Copy(m_params.HiddenVectorUserParsed, m_world.HiddenVectorOutput.Host, m_params.HiddenVectorLength);
            }
            m_world.HiddenVectorOutput.SafeCopyToDevice();
        }

        private void EraseGameBoard()
        {
            ResetHiddenVector();

            m_world.GuessCountOutput.Fill(0.0f);

            m_world.GuessHistoryOutput.Fill(0.0f);
            m_world.GuessEvaluationHistoryOutput.Fill(0.0f);

            // erase visual output not necessary - done by RenderTask
        }

        /// <summary>
        /// Called from InitTask
        /// </summary>
        public void Reset()
        {
            EraseGameBoard();
            m_world.WorldEventOutput.Fill(0.0f);
        }

        private void ResetWon()
        {
            EraseGameBoard();
            m_world.WorldEventOutput.Fill(1.0f);
        }

        private void ResetLost()
        {
            EraseGameBoard();
            m_world.WorldEventOutput.Fill(-1.0f);
        }

        /// <summary>
        /// Returns the number of bulls (direct hits) and cows (good color, but wrong position) in the guess.
        /// The vector in guess is compared with the vector in hidden.
        /// </summary>
        private void GetEvaluation(float[] guess, float[] hidden, out int bulls, out int cows)
        {
            bulls = 0;
            cows = 0;

            Dictionary<int, int> colorHistoHidden = new Dictionary<int, int>();
            for (int i = 0; i < hidden.Length; i++)
            {
                int key = (int)Math.Round(hidden[i]);
                if (!colorHistoHidden.ContainsKey(key))
                    colorHistoHidden[key] = 1;
                else
                    colorHistoHidden[key]++;
            }


            // first count the number of bulls
            for (int i = 0; i < guess.Length; i++)
            {
                if (Math.Round(guess[i]) == Math.Round(hidden[i]))
                {
                    bulls++;
                    // do not report the bull as a cow!
                    int key = (int)Math.Round(guess[i]);
                    colorHistoHidden[key]--;
                }
            }

            // now count the number of cows
            for (int i = 0; i < guess.Length; i++)
            {
                if (Math.Round(guess[i]) != Math.Round(hidden[i])) // not a bull
                {
                    int key = (int)Math.Round(guess[i]);
                    if(key >= m_params.NumberOfColors || key < 0)
                    {
                        MyLog.WARNING.WriteLine("Mastermind world: the submitted guess contains an out-of-range color: {0}, range: <{1}-{2}>", 
                            guess[i], 0, m_params.NumberOfColors-1);
                    }
                    if(colorHistoHidden.ContainsKey(key))
                    {
                        if(colorHistoHidden[key] > 0)
                        {
                            cows++;
                            colorHistoHidden[key]--;
                        }
                    }
                }
            }

            // returns bulls and cows as out parameters
        }

        /// <summary>
        /// Adds the newest guess to the history of guesses.
        /// Adds the evaluation of the newest guess to the history of evaluations.
        /// </summary>
        /// <param name="guess">the guess to add</param>
        /// <param name="oldGuessCount">the number of guesses and evaluations already stored in history.</param>
        private void AddGuess(float[] guess, int oldGuessCount)
        {
            int guessesOutputOffset = m_params.HiddenVectorLength * oldGuessCount;
            m_world.GuessHistoryOutput.SafeCopyToHost(guessesOutputOffset, m_params.HiddenVectorLength);
            Array.Copy(guess, 0, m_world.GuessHistoryOutput.Host, guessesOutputOffset, guess.Length);
            m_world.GuessHistoryOutput.SafeCopyToDevice(guessesOutputOffset, m_params.HiddenVectorLength);

            int bulls = 0, cows = 0;
            int evaluationsOutputOffset = oldGuessCount * MyMastermindWorld.EVALUATION_ITEM_LENGTH;
            GetEvaluation(guess, m_world.HiddenVectorOutput.Host, out bulls, out cows);
            m_world.GuessEvaluationHistoryOutput.SafeCopyToHost(evaluationsOutputOffset, MyMastermindWorld.EVALUATION_ITEM_LENGTH);
            m_world.GuessEvaluationHistoryOutput.Host[evaluationsOutputOffset] = bulls;
            m_world.GuessEvaluationHistoryOutput.Host[evaluationsOutputOffset + 1] = cows;
            m_world.GuessEvaluationHistoryOutput.SafeCopyToDevice(evaluationsOutputOffset, MyMastermindWorld.EVALUATION_ITEM_LENGTH);
        }

        /// <summary>
        /// // The agent has submitted a guess and confirmed it. Compute next state of the world.
        /// </summary>
        public void Step()
        {
            // 0) Get the guess.
            float[] guess = new float[m_params.HiddenVectorLength];
            if (m_world.GuessInput != null)
            {
                m_world.GuessInput.SafeCopyToHost();
                Array.Copy(m_world.GuessInput.Host, guess, guess.Length);
            }
            else
            {
                Array.Clear(guess, 0, guess.Length);
            }

            // 1) Check if the guess is the hidden vector. If it is, output a reward in the WorldEventOutput and reset 
            //    the game.
            m_world.HiddenVectorOutput.SafeCopyToHost();
            bool equal = true;
            for (int i = 0; i < guess.Length; i++)
            {
                equal = Math.Round(guess[i]) == Math.Round(m_world.HiddenVectorOutput.Host[i]);
                if (!equal)
                    break;
            }
            if(equal)
            {
                ResetWon();
                return;
            }
            
            // 2) Otherwise check if the number of guesses has been reached. If so, reset the game
            m_world.GuessCountOutput.SafeCopyToHost();
            int oldGuessCount = (int)Math.Round(m_world.GuessCountOutput.Host[0]);
            m_world.GuessCountOutput.Host[0] += 1.0f; // one more guess was made
            if(m_world.GuessCountOutput.Host[0] >= m_params.NumberOfGuesses)
            {
                ResetLost();
                return;
            }
            m_world.GuessCountOutput.SafeCopyToDevice();
            
            // 3) Otherwise add the guess to the history of the guesses. Compute an evaluation of the guess and add it
            //    to the history of evaluations.
            AddGuess(guess, oldGuessCount);
        }
        
    }
}

