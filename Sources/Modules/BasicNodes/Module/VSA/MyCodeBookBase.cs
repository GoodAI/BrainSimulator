using GoodAI.Core.Memory;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.VSA
{
    public enum MyCodeVector
    {
        Empty,
        Goal,
        Subject,
        Memory,
        Visual,

        Position,
        Left,
        Center,
        Right,
        Top,
        Middle,
        Bottom,
        DirX,
        DirY,
        OriginX,
        OriginY,
        Velocity,

        Size,
        Small,
        Medium,
        Big,

        Color,

        Extra,
        NegDirX,
        NegDirY,

        Ball,
        Paddle
    }


    public abstract class MyCodeBookBase : MyRandomPool
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }


        #region MyRandomPool overrides

        protected override string GlobalVariableName
        {
            get { return "CODE_VECTORS_" + SymbolSize; }
        }

        protected override int PatternCount
        {
            get { return typeof(MyCodeVector).GetEnumValues().Length; }
        }

        public override int Seed
        {
            get { return 1234; }
        }

        protected override float[] GenerateRandomVectors()
        {
            float[] pool = base.GenerateRandomVectors();

            for (int i = 0; i < SymbolSize; i++)
            {
                pool[i] = 0;
            }

            return pool;
        }

        #endregion
    }
}
