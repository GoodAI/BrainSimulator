using System;
using System.Diagnostics.Contracts;

namespace Utils.VRageRIP.Lib.Extensions
{
    public class ArrayCreator
    {
        public static T CreateJaggedArray<T>(params int[] lengths)
        {
            return (T)InitializeJaggedArray(null, 0, lengths);
        }

        static object InitializeJaggedArray(Type type, int index, int[] lengths)
        {
            if (type == null)
                throw new ArgumentNullException("type");
            Contract.EndContractBlock();
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
