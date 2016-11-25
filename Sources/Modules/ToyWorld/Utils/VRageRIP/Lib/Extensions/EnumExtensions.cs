using System;

namespace Utils.VRageRIP.Lib.Extensions
{
    public static class EnumExtensions
    {
        public static TEnumFlags FlipEnumFlag<TEnumFlags>(this TEnumFlags value, TEnumFlags flag)
            where TEnumFlags : struct, IConvertible, IComparable, IFormattable // Should be an enum
        {
            int valueInt = Convert.ToInt32(value);
            int flagInt = Convert.ToInt32(flag);
            valueInt ^= flagInt;
            return (TEnumFlags)Enum.Parse(typeof(TEnumFlags), valueInt.ToString());
        }
    }
}
