using System;
using System.Linq;
using System.Reflection;

namespace GoodAI.Core.Utils
{
    public static class MyAttributeExtension
    {
        public static TValue GetAttributeProperty<TAttribute, TValue>(this MemberInfo type, Func<TAttribute, TValue> valueSelector)
            where TAttribute : Attribute
        {
            var att = type.GetCustomAttributes(
                typeof(TAttribute), true
            ).FirstOrDefault() as TAttribute;
            if (att != null)
            {
                return valueSelector(att);
            }
            return default(TValue);
        }

        public static TValue GetAttributeProperty<TAttribute, TValue>(this Enum enumVal, Func<TAttribute, TValue> valueSelector)
            where TAttribute : Attribute
        {
            var type = enumVal.GetType();
            var memInfo = type.GetMember(enumVal.ToString());
            var att =  memInfo[0].GetCustomAttributes(
                typeof(TAttribute), true
            ).FirstOrDefault() as TAttribute;
            if (att != null)
            {
                return valueSelector(att);
            }
            return default(TValue);
        }

    }
}
