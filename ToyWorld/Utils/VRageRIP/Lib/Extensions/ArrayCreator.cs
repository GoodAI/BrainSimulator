using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils.VRageRIP.Lib.Extensions
{
    public class ArrayCreator
    {
        public static T CreateJaggedArray<T>(params int[] lengths)
        {
            return (T)InitializeJaggedArray(typeof(T).GetElementType(), 0, lengths);
        }

        static object InitializeJaggedArray(Type type, int index, int[] lengths)
        {
            MyContract.Requires<ArgumentNullException>(type != null, "Type cannot be null!");

            Array array = Array.CreateInstance(type, lengths[index]);
            Type elementType = type.GetElementType();

            if (elementType == null)
                return array;

            for (int i = 0; i < lengths[index]; i++)
            {
                array.SetValue(
                    InitializeJaggedArray(elementType, index + 1, lengths), i);
            }

            return array;
        }
    }
}
