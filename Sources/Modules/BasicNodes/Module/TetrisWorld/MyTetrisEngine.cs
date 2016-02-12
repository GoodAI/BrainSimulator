using ManagedCuda.BasicTypes;
using System;
using System.Diagnostics;
using System.Linq;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using GoodAI.Core.Utils;
using GoodAI.Core.Memory;

namespace GoodAI.Modules.TetrisWorld
{
    /// <summary>
    /// Holds parameters of the game world that are used both by the world and by the engine
    /// </summary>
    public sealed class WorldEngineParams
    {
        public int ClearedLinesPerLevel;
        public int AlmostFullLinesAtStart;
        public int WaitStepsPerFall;
    }

    public enum BrickType
    {
        None = 0,
        I = 1,
        J = 2,
        L = 3,
        O = 4,
        S = 5,
        T = 6,
        Z = 7,
        Preset = 8
    }

    /// <summary>
    /// Values of the enum constants have meaning, do not change.
    /// </summary>
    public enum TetrominoRotation
    {
        None = 0,
        Left = 1,
        UpsideDown = 2,
        Right = 3,
    }

    /// <summary>
    /// Holds the state of the brick area during the game. 
    /// Updates the state of the brick area after a tetromino lands.
    /// </summary>
    public class TetrisGameBoard
    {
        private BrickType[,] m_gameBoardState;
        private int[] m_horizon;
        private int m_columns;
        private int m_rows;

        public BrickType[,] GameBoardState { get { return m_gameBoardState; } }
        public int[] Horizon {
            get
            {
                return m_horizon;
            }
            set
            {
                m_horizon = value;
            }
        }

        public TetrisGameBoard(int columns, int rows)
        {
            m_gameBoardState = new BrickType[rows, columns];
            m_columns = columns;
            m_rows = rows;
            m_horizon = new int[m_columns];
        }

        public TetrisGameBoard(TetrisGameBoard source)
        {
            m_rows = source.m_rows;
            m_columns = source.m_columns;
            m_horizon = new int[m_columns];
            m_gameBoardState = new BrickType[source.m_rows, source.m_columns];
            Array.Copy(source.m_gameBoardState, m_gameBoardState, m_columns * m_rows);
            CalculateHorizon();
        }

        public TetrisGameBoard(int[] horizon, int rows = 20)
        {
            m_rows = rows;
            m_columns = horizon.Count();            
            m_gameBoardState = new BrickType[m_rows, m_columns];
            SetHorizon(horizon);
        }

        /// <summary>
        /// Erases filled lines from the game board.
        /// </summary>
        /// <returns>number of lines erased</returns>
        public int EraseFullLines()
        {
            int erasedLines = 0;
            for (int iRow = 0; iRow < m_rows; iRow++)
            {
                bool isFullLine = true;
                for (int iCol = 0; iCol < m_columns; iCol++)
                {
                    if (m_gameBoardState[iRow, iCol] == BrickType.None)
                    {
                        isFullLine = false;
                        break; // try next row
                    }
                }
                if (isFullLine)
                {
                    for (int iTargetRow = iRow; iTargetRow > 0; iTargetRow--)
                    {
                        for (int iTargetCol = 0; iTargetCol < m_columns; iTargetCol++)
                        {
                            m_gameBoardState[iTargetRow, iTargetCol] = m_gameBoardState[iTargetRow - 1, iTargetCol];
                        }
                    }
                    if (erasedLines == 0)
                    {
                        // handle top row
                        for (int iTargetCol = 0; iTargetCol < m_columns; iTargetCol++)
                        {
                            m_gameBoardState[0, iTargetCol] = BrickType.None;
                        }
                    }
                    erasedLines++;
                }
            }
            return erasedLines;
        }

        /// <summary>
        ///    Puts the tetromino into the brick area.
        ///    Called when the engine prepares data for structured and visual output of the brick area or when
        ///    a falling tetromino hits the bottom.
        /// </summary>
        public void MergeTetrominoWithGameBoard(MovingTetromino tetromino)
        {
            Debug.Assert(tetromino != null & tetromino.DescriptiveGrid != null);

            BrickType[,] tetrominoGrid = tetromino.DescriptiveGrid;

            // put the moving tetromino into the game board
            for (int iRow = 0; iRow < tetromino.Height; iRow++)
            {
                for (int iColumn = 0; iColumn < tetromino.Width; iColumn++)
                {
                    if (tetrominoGrid[iRow, iColumn] != BrickType.None)
                    {
                        m_gameBoardState[tetromino.Row + iRow, tetromino.Column + iColumn] = tetrominoGrid[iRow, iColumn];
                    }
                }
            }
            CalculateHorizon();
        }

        /// <summary>
        /// Can move the tetromino to within the horizontal boundaries of the game board (left,right boundaries only). 
        /// </summary>
        /// <param name="tetrominoGrid"></param>
        /// <param name="column">Column in which the left top cell of the descriptive grid is. Top row has index 0.</param>
        /// <param name="row">Row in which the left top cell of the descriptive grid is. Leftmost column has index 0.</param>
        /// <returns>Returns false if the tetromino starts to intersect the bottom.</returns>
        public bool FitTetrominoToGameBoard(BrickType[,] tetrominoGrid, ref int column, ref int row)
        {
            // first move the tetromino within the boundaries:
            for (int iRow = 0; iRow < tetrominoGrid.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < tetrominoGrid.GetLength(1); iCol++)
                {
                    if (tetrominoGrid[iRow, iCol] != BrickType.None)
                    {
                        if (row + iRow < 0)
                        {
                            // not a problem, does not happen
                        }
                        if (row + iRow >= m_rows)
                        {
                            return false; // bottom was intersected
                        }
                        if (column + iCol < 0)
                        {
                            column = -iCol;
                        }
                        if (column + iCol >= m_columns)
                        {
                            column = m_columns - iCol - 1;
                        }
                    }
                }
            }
            return true;
        }

        public bool IsTetrominoInCollision(BrickType[,] tetrominoGrid, int column, int row)
        {
            // check for collisions between the tetromino and the game board's bricks
            for (int iRow = 0; iRow < tetrominoGrid.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < tetrominoGrid.GetLength(1); iCol++)
                {
                    if (tetrominoGrid[iRow, iCol] != BrickType.None &&
                        m_gameBoardState[iRow + row, iCol + column] != BrickType.None)
                        return true; // collision
                }
            }
            return false;
        }

        private void MakeBoardByHorizon()
        {
            int min = m_horizon.Min();
            BrickType brick = BrickType.None;
            for (int iRow = 0; iRow < m_rows; iRow++)
                for (int iCol = 0; iCol < m_columns; iCol++)
                {
                    if (iRow < m_rows - 1 - (m_horizon[iCol] - min))
                        brick = BrickType.None;
                    else
                        brick = BrickType.Preset;
                    m_gameBoardState[iRow, iCol] = brick;
                }
        }

        private void CalculateHorizon()
        {
            int iRow = 0;
            for (int iCol = 0; iCol < m_columns; iCol++)
            { 
                iRow = m_rows - 1;
                while (iRow >= 0 && m_gameBoardState[iRow, iCol] == BrickType.None)
                {
                    iRow--;
                }
                m_horizon[iCol] = iRow;
            }
            m_horizon = Array.ConvertAll(m_horizon, x => -(x - iRow));  //last brick will be at level 0 (just for simplicity), holes are negative, hills positive
        }

        public void SetHorizon(int[] horizon)
        {
            Debug.Assert(horizon.Count() == m_columns, "You need to set horizon of proper width");
            Array.Copy(horizon, m_horizon, m_columns);
            MakeBoardByHorizon();
        }

        public bool DoesBrickFitAtIdx(int[] brick, int idx)
        {
            bool result = false;
            int bWidth = brick.Count();
            int[] tmp = new int[bWidth];
            for (int bi = 0; bi < bWidth && idx + bi < m_columns; bi++)
            {
                tmp[bi] = m_horizon[idx + bi] - brick[bi]; // add a brick to horizon 
            }
            if (tmp.Distinct().Count() == 1) // if result levels are same, brick fits
                result = true;
            return result;        
        }

        public bool DoesBrickFitToHorizon(int[] brick)
        {
            bool result = false;
            int idx = 0;
            int bWidth = brick.Count();
            while (!result && idx < m_columns - bWidth)
            {
                result = DoesBrickFitAtIdx(brick, idx);
                idx++;
            }
            return result;
        }

    }

    public class MovingTetrominoFactory
    {
        public static MovingTetromino CreateMovingTetromino(TetrisGameBoard gameBoard, BrickType brick)
        {
            MovingTetromino inst = null;
            switch(brick)
            {
                case BrickType.J:
                    {
                        inst = new MovingTetrominoJ(gameBoard);
                        break;
                    }
                case BrickType.L:
                    {
                        inst = new MovingTetrominoL(gameBoard);
                        break;
                    }
                case BrickType.O:
                    {
                        inst = new MovingTetrominoO(gameBoard);
                        break;
                    }
                case BrickType.S:
                    {
                        inst = new MovingTetrominoS(gameBoard);
                        break;
                    }
                case BrickType.T:
                    {
                        inst = new MovingTetrominoT(gameBoard);
                        break;
                    }
                case BrickType.Z:
                    {
                        inst = new MovingTetrominoZ(gameBoard);
                        break;
                    }
                default:
                case BrickType.I:
                    {
                        inst = new MovingTetrominoI(gameBoard);
                        break;
                    }
            }
            inst.InitializeDescriptiveGrid();
            return inst;
        }
    }

    /// <summary>
    /// Class that represents the tetromino that is falling down. Specializations of the class define the different 
    /// tetromino shapes.
    /// </summary>
    public abstract class MovingTetromino
    {
        public int Column { get; protected set; } // position of top left corner in the game board
        public int Row { get; protected set; }    // position of top left corner in the game board
        public int Width { get { return 4; } }
        public int Height { get { return 4; } }
        public BrickType[,] DescriptiveGrid { get; protected set; } // do not change the returned grid!
        public abstract BrickType TetrominoBrickType { get; }
        protected abstract BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation);
        public abstract int[] GetRotatedLowerHorizon(TetrominoRotation rotation);
        public TetrominoRotation Rotation { get; protected set; }

        protected TetrisGameBoard m_gameBoard;

        public MovingTetromino(TetrisGameBoard gameBoard)
        {
            m_gameBoard = gameBoard;
            Rotation = TetrominoRotation.None;
            Column = 3; // starts in the middle
            Row = 0;
            DescriptiveGrid = null;
        }

        public void InitializeDescriptiveGrid()
        {
            DescriptiveGrid = GetRotatedDescriptiveGrid(Rotation);
        }

        public bool IsOverlappingGameBoardBricks()
        {
            return m_gameBoard.IsTetrominoInCollision(DescriptiveGrid, Column, Row);
        }

        /// <summary>
        /// tries to fit the provided tetromino (represented by its descriptiveGrid) into the game board
        /// </summary>
        /// <param name="descriptiveGrid"></param>
        /// <param name="column"></param>
        /// <param name="row"></param>
        /// <param name="rotation"></param>
        /// <returns>true if the provided tetromino could be fit into the game board</returns>
        protected bool TryMovement(BrickType[,] descriptiveGrid, int column, int row, TetrominoRotation rotation)
        {
            bool bottomHit = !m_gameBoard.FitTetrominoToGameBoard(descriptiveGrid, ref column, ref row);
            if (!bottomHit && !m_gameBoard.IsTetrominoInCollision(descriptiveGrid, column, row))
            {
                Rotation = rotation;
                DescriptiveGrid = descriptiveGrid;
                Column = column;
                Row = row;
                return true;
            }
            return false;
        }

        public bool TryRotateLeft()
        {
            TetrominoRotation rot = Rotation; 
            rot = (TetrominoRotation)((int)(rot+1) % 4);
            BrickType[,] descriptiveGrid = GetRotatedDescriptiveGrid(rot);
            int column = Column, row = Row;
            return TryMovement(descriptiveGrid, column, row, rot);
        }

        public bool TryRotateRight()
        {
            TetrominoRotation rot = Rotation;
            rot = (TetrominoRotation)((int)(rot + 3) % 4); // 3 == -1 mod 4
            BrickType[,] descriptiveGrid = GetRotatedDescriptiveGrid(rot);
            int column = Column, row = Row;
            return TryMovement(descriptiveGrid, column, row, rot);
        }

        public bool TryMoveDown()
        {
            TetrominoRotation rot = Rotation;
            BrickType[,] descriptiveGrid = GetRotatedDescriptiveGrid(rot);
            int column = Column, row = Row + 1;
            return TryMovement(descriptiveGrid, column, row, rot);
        }

        public bool TryMoveLeft()
        {
            TetrominoRotation rot = Rotation;
            BrickType[,] descriptiveGrid = GetRotatedDescriptiveGrid(rot);
            int column = Column - 1, row = Row;
            return TryMovement(descriptiveGrid, column, row, rot);
        }

        public bool TryMoveRight()
        {
            TetrominoRotation rot = Rotation;
            BrickType[,] descriptiveGrid = GetRotatedDescriptiveGrid(rot);
            int column = Column + 1, row = Row;
            return TryMovement(descriptiveGrid, column, row, rot);
        }
    }

    public class MovingTetrominoI : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.I,    BrickType.I,    BrickType.I,    BrickType.I    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotL = { { BrickType.None, BrickType.I,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.I,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.I,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.I,    BrickType.None, BrickType.None }};

        public MovingTetrominoI(TetrisGameBoard gameBoard) : base(gameBoard) {}

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.I; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {
            
            switch(rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                        return gridRot0;
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                        return gridRotL;
            }
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation) 
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                    return new int[] {0, 0, 0, 0};
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                    return new int[] { 0 };
            }        
        }

    }

    public class MovingTetrominoJ : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.J,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.J,    BrickType.J,    BrickType.J    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotL = { { BrickType.None, BrickType.None, BrickType.J,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.J,    BrickType.None },
                                         { BrickType.None, BrickType.J,    BrickType.J,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotR = { { BrickType.None, BrickType.J,    BrickType.J,    BrickType.None },
                                         { BrickType.None, BrickType.J,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.J,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotU = { { BrickType.None, BrickType.J,    BrickType.J,    BrickType.J    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.J    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        public MovingTetrominoJ(TetrisGameBoard gameBoard) : base(gameBoard) { }

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.J; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {

            switch (rotation)
            {
                case TetrominoRotation.None:
                    return gridRot0;
                case TetrominoRotation.UpsideDown:
                    return gridRotU;
                case TetrominoRotation.Left:
                    return gridRotL;
                case TetrominoRotation.Right:
                default:
                    return gridRotR;
            }
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation)
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                    return new int[] { 0, 0, 0};
                case TetrominoRotation.UpsideDown:
                    return new int[] { 1, 1, 0 };
                case TetrominoRotation.Left:
                    return new int[] { 0, 0 };
                case TetrominoRotation.Right:
                default:
                    return new int[] { 0, 2 };
            }
        }

    }

    public class MovingTetrominoL : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.None, BrickType.None, BrickType.L    },
                                         { BrickType.None, BrickType.L,    BrickType.L,    BrickType.L    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotL = { { BrickType.None, BrickType.L,    BrickType.L,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.L,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.L,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotR = { { BrickType.None, BrickType.L,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.L,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.L,    BrickType.L,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotU = { { BrickType.None, BrickType.L,    BrickType.L,    BrickType.L    },
                                         { BrickType.None, BrickType.L,    BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        public MovingTetrominoL(TetrisGameBoard gameBoard) : base(gameBoard) { }

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.L; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {

            switch (rotation)
            {
                case TetrominoRotation.None:
                    return gridRot0;
                case TetrominoRotation.UpsideDown:
                    return gridRotU;
                case TetrominoRotation.Left:
                    return gridRotL;
                case TetrominoRotation.Right:
                default:
                    return gridRotR;
            }
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation)
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                    return new int[] { 0, 0, 0 };
                case TetrominoRotation.UpsideDown:
                    return new int[] { 0, 1, 1 };
                case TetrominoRotation.Left:
                    return new int[] { 2, 0 };
                case TetrominoRotation.Right:
                default:
                    return new int[] { 0, 0 };
            }
        }

    }

    public class MovingTetrominoO : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.O,    BrickType.O,    BrickType.None },
                                         { BrickType.None, BrickType.O,    BrickType.O,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        public MovingTetrominoO(TetrisGameBoard gameBoard) : base(gameBoard) { }

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.O; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {

            return gridRot0;
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation)
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                    return new int[] { 0, 0 };
            }
        }
    }

    public class MovingTetrominoS : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.None, BrickType.S,    BrickType.S    },
                                         { BrickType.None, BrickType.S,    BrickType.S,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRot90 = { { BrickType.None, BrickType.S,    BrickType.None, BrickType.None },
                                          { BrickType.None, BrickType.S,    BrickType.S,    BrickType.None },
                                          { BrickType.None, BrickType.None, BrickType.S,    BrickType.None },
                                          { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        public MovingTetrominoS(TetrisGameBoard gameBoard) : base(gameBoard) { }

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.S; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {

            switch (rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                        return gridRot0;
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                        return gridRot90;
            }
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation)
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                    return new int[] { 0, 0, 1 };
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                    return new int[] { 1, 0 };
            }
        }

    }

    public class MovingTetrominoT : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.None, BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.T,    BrickType.T,    BrickType.T    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotR = { { BrickType.None, BrickType.None, BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.T,    BrickType.T    },
                                         { BrickType.None, BrickType.None, BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotL = { { BrickType.None, BrickType.None, BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.T,    BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRotU = { { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.T,    BrickType.T,    BrickType.T    },
                                         { BrickType.None, BrickType.None, BrickType.T,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        public MovingTetrominoT(TetrisGameBoard gameBoard) : base(gameBoard) { }

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.T; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {

            switch (rotation)
            {
                case TetrominoRotation.None:
                    return gridRot0;
                case TetrominoRotation.UpsideDown:
                    return gridRotU;
                case TetrominoRotation.Left:
                    return gridRotL;
                case TetrominoRotation.Right:
                default:
                    return gridRotR;
            }
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation)
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                    return new int[] { 0, 0, 0 };
                case TetrominoRotation.UpsideDown:
                    return new int[] { 1, 0, 1 };
                case TetrominoRotation.Left:
                    return new int[] { 1, 0 };
                case TetrominoRotation.Right:
                default:
                    return new int[] { 0, 1 };
            }
        }

    }

    public class MovingTetrominoZ : MovingTetromino
    {
        static BrickType[,] gridRot0 = { { BrickType.None, BrickType.Z,    BrickType.Z,    BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.Z,    BrickType.Z    },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None },
                                         { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        static BrickType[,] gridRot90 = { { BrickType.None, BrickType.None, BrickType.None, BrickType.Z   },
                                          { BrickType.None, BrickType.None, BrickType.Z,    BrickType.Z   },
                                          { BrickType.None, BrickType.None, BrickType.Z,    BrickType.None },
                                          { BrickType.None, BrickType.None, BrickType.None, BrickType.None }};

        public MovingTetrominoZ(TetrisGameBoard gameBoard) : base(gameBoard) { }

        public override BrickType TetrominoBrickType
        {
            get { return BrickType.Z; }
        }

        protected override BrickType[,] GetRotatedDescriptiveGrid(TetrominoRotation rotation)
        {

            switch (rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                        return gridRot0;
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                        return gridRot90;
            }
        }

        public override int[] GetRotatedLowerHorizon(TetrominoRotation rotation)
        {
            switch (rotation)
            {
                case TetrominoRotation.None:
                case TetrominoRotation.UpsideDown:
                    return new int[] { 1, 0, 0 };
                case TetrominoRotation.Left:
                case TetrominoRotation.Right:
                default:
                    return new int[] { 0, 1 };
            }
        }

    }

    /// <summary>
    /// Holds state and defines the behavior of all objects in the world.
    /// </summary>
    public sealed class TetrisWorldEngine
    {
        public bool IsMerging { get; private set; }
        public bool CleanMatch { get; private set; }

        private WorldEngineParams m_params;
        private TetrisWorld m_world;
        private TetrisGameBoard m_gameBoard;
        private MovingTetromino m_tetromino;

        private Random m_rndGen;
        private int m_stepsFromLastDrop;
        private bool m_shouldSpawnTetromino;
        private int m_level;
        private int m_score;
        private int m_scoreDelta;
        private int m_totalErasedLines;
        private MovingTetromino m_nextTetromino;

        public TetrisWorldEngine(TetrisWorld world, WorldEngineParams pars)
        {
            m_params = pars;
            m_world = world;
            m_rndGen = new Random();
        }

        private void ReinitGameBoard()
        {
            m_world.BrickAreaOutput.Fill((float)BrickType.None);

            m_world.HintAreaOutput.Fill((float)BrickType.None);

            m_world.NextBrickNumberOutput.Fill(0.0f);

            m_world.ScoreOutput.Fill(0.0f);

            m_world.ScoreDeltaOutput.Fill(0.0f);

            m_world.LevelOutput.Fill(0.0f);

            // erase visual output not necessary - done by RenderTask

            m_gameBoard = new TetrisGameBoard(m_world.BrickAreaColumns, m_world.BrickAreaRows);

            // add almost full lines
            for(int iRow = 0; iRow < m_params.AlmostFullLinesAtStart; iRow++)
            {
                int iRowIndex = m_world.BrickAreaRows - 1 - iRow;
                for(int iCol = 0; iCol < m_world.BrickAreaColumns; iCol++)
                {
                    m_gameBoard.GameBoardState[iRowIndex,iCol] = BrickType.Preset;
                }
                int holeIndex = m_rndGen.Next(0, 10);
                m_gameBoard.GameBoardState[iRowIndex, holeIndex] = BrickType.None;
            }

            m_stepsFromLastDrop = 0;
            m_level = 0;
            m_score = 0;
            m_scoreDelta = 0;
            m_shouldSpawnTetromino = true;
            m_tetromino = null;
            m_totalErasedLines = 0;
            PrepareNextTetromino();
        }

        /// <summary>
        /// Called from InitTask
        /// </summary>
        public void Reset()
        {
            ReinitGameBoard();
            m_world.WorldEventOutput.Fill(0.0f);
        }

        public void ResetToRandomHorizon(int range = 2)
        {
            Reset();
            Random rnd = new Random();
            int[] horizon = Enumerable
                .Repeat(0, m_world.BrickAreaColumns)
                .Select(i => rnd.Next(-range, range))
                .ToArray();
            m_gameBoard.SetHorizon(horizon);
        }

        public bool CanMatchAnyRotation()
        {
            bool match = CanMatch();
            match = match || CanMatch(TetrominoRotation.Left);
            match = match || CanMatch(TetrominoRotation.Right);
            match = match || CanMatch(TetrominoRotation.UpsideDown);
            return match;
        }

        public bool CanMatch(TetrominoRotation rotation = TetrominoRotation.None)
        {
            return m_gameBoard.DoesBrickFitToHorizon(m_tetromino.GetRotatedLowerHorizon(rotation));
        }

        private void ResetLost()
        {
            ReinitGameBoard();
            m_world.WorldEventOutput.Fill(-1.0f);
        }

        private int GetWaitStepsPerDrop()
        {
            return Math.Max(0, m_params.WaitStepsPerFall - m_level);
        }

        private void PrepareNextTetromino()
        {
            BrickType next = (BrickType)(m_rndGen.Next(0, 7) + 1);
            m_nextTetromino = MovingTetrominoFactory.CreateMovingTetromino(m_gameBoard, next);
        }

        // return false if the spawned tetromino intersects bricks already present on the game board
        private bool SpawnTetromino()
        {
            m_tetromino = m_nextTetromino;
            PrepareNextTetromino();
            return !m_tetromino.IsOverlappingGameBoardBricks();
        }

        /// <summary>
        /// Compute next state of the world. Complete game logic for one step.
        /// </summary>
        public void Step(TetrisWorld.ActionInputType input)
        {
            IsMerging = false;

            if (m_shouldSpawnTetromino)
            {
                m_shouldSpawnTetromino = false;
                if(!SpawnTetromino())
                {
                    // game over
                    ResetLost();
                    return;
                }
            }

            bool isDrop = false;
            if (m_stepsFromLastDrop >= GetWaitStepsPerDrop())
            {
                isDrop = true;
            }

            switch(input)
            {
                case TetrisWorld.ActionInputType.MoveDown:
                    {
                        isDrop = true;
                        break;
                    }
                case TetrisWorld.ActionInputType.MoveLeft:
                    {
                        m_tetromino.TryMoveLeft();
                        break;
                    }
                case TetrisWorld.ActionInputType.MoveRight:
                    {
                        m_tetromino.TryMoveRight();
                        break;
                    }
                case TetrisWorld.ActionInputType.RotateLeft:
                    {
                        m_tetromino.TryRotateLeft();
                        break;
                    }
                case TetrisWorld.ActionInputType.RotateRight:
                    {
                        m_tetromino.TryRotateRight();
                        break;
                    }
                default:
                case TetrisWorld.ActionInputType.NoAction:
                    {
                        break;
                    }
            }

            m_scoreDelta = 0;

            if (isDrop)
            {
                m_stepsFromLastDrop = 0;
                if(!m_tetromino.TryMoveDown())
                {
                    IsMerging = true;
                    CalculateMerge();
                    m_gameBoard.MergeTetrominoWithGameBoard(m_tetromino);
                    int erasedLines = m_gameBoard.EraseFullLines(); // most often returns 0
                    m_scoreDelta = 100 * erasedLines * erasedLines;
                    m_score += m_scoreDelta;
                    m_totalErasedLines += erasedLines;
                    m_level = m_totalErasedLines / Math.Max(1,m_params.ClearedLinesPerLevel);
                    
                    m_tetromino = null;
                    m_shouldSpawnTetromino = true;
                }
                
            }
            else
            {
                m_stepsFromLastDrop++;
            }

            FillWorldState();
        }

        private void CalculateMerge()
        {
            // calculate real position of leftmost brick piece
            var col = m_tetromino.Column + 1;
            switch (m_tetromino.TetrominoBrickType)
            {
                case BrickType.T:
                    if (m_tetromino.Rotation == TetrominoRotation.Right)
                        col++;
                    break;
                case BrickType.I:
                    if (m_tetromino.Rotation == TetrominoRotation.None || m_tetromino.Rotation == TetrominoRotation.UpsideDown)
                        col--;
                    break;
                case BrickType.Z:
                    if (m_tetromino.Rotation == TetrominoRotation.Right || m_tetromino.Rotation == TetrominoRotation.Left)
                        col++;
                    break;
            }
            CleanMatch = m_gameBoard.DoesBrickFitAtIdx(m_tetromino.GetRotatedLowerHorizon(m_tetromino.Rotation), col);
        }

        /// <summary>
        /// copies game state to world's memory blocks
        /// </summary>
        private void FillWorldState()
        {
            FillBrickAreaOutput();

            FillHintAreaOutput();

            m_world.NextBrickNumberOutput.Host[0] = (float)m_nextTetromino.TetrominoBrickType;
            m_world.NextBrickNumberOutput.SafeCopyToDevice();

            m_world.ScoreOutput.Host[0] = (float)m_score;
            m_world.ScoreOutput.SafeCopyToDevice();

            m_world.ScoreDeltaOutput.Host[0] = (float)m_scoreDelta;
            m_world.ScoreDeltaOutput.SafeCopyToDevice();

            m_world.WorldEventOutput.Host[0] = m_scoreDelta > 0 ? m_scoreDelta/100.0f : 0.0f;
            m_world.WorldEventOutput.SafeCopyToDevice();

            m_world.LevelOutput.Host[0] = (float)m_level;
            m_world.LevelOutput.SafeCopyToDevice();
            
            // visual output is updated by the RenderGame task
        }

        private void FillHintAreaOutput()
        {
            BrickType[,] descriptiveGrid = m_nextTetromino.DescriptiveGrid;
            int index = 0;
            for (int iRow = 0; iRow < descriptiveGrid.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < descriptiveGrid.GetLength(1); iCol++)
                {
                    m_world.HintAreaOutput.Host[index] = (float)descriptiveGrid[iRow, iCol];
                    index++;
                }
            }
            m_world.HintAreaOutput.SafeCopyToDevice();
        }

        private void FillBrickAreaOutput()
        {
            TetrisGameBoard gameBoardMerged = new TetrisGameBoard(m_gameBoard);
            if (m_tetromino != null)
            {
                gameBoardMerged.MergeTetrominoWithGameBoard(m_tetromino);
            }

            BrickType[,] gameBoardGrid = gameBoardMerged.GameBoardState;
            int index = 0;
            for (int iRow = 0; iRow < gameBoardGrid.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < gameBoardGrid.GetLength(1); iCol++)
                {
                    m_world.BrickAreaOutput.Host[index] = (float)gameBoardGrid[iRow, iCol];
                    index++;
                }
            }
            m_world.BrickAreaOutput.SafeCopyToDevice();
        }
    }
}

