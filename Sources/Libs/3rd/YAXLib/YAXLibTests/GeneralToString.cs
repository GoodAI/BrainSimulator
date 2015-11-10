// Copyright 2009 - 2010 Sina Iravanian - <sina@sinairv.com>
//
// This source file(s) may be redistributed, altered and customized
// by any means PROVIDING the authors name and all copyright
// notices remain intact.
// THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED. USE IT AT YOUR OWN RISK. THE AUTHOR ACCEPTS NO
// LIABILITY FOR ANY DATA DAMAGE/LOSS THAT THIS PRODUCT MAY CAUSE.
//-----------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Collections;
using YAXLib;

namespace YAXLibTests
{
    public class GeneralToStringProvider
    {
        public static string GeneralToString(object o)
        {
            return GeneralToString(o, 0);
        }

        private static string CollectionToString(object collectionInstance, string propName, int layer)
        {
            //object collectionInstance = prop.GetValue(o, null);
            StringBuilder sb = new StringBuilder();
            if (collectionInstance == null)
            {
                if(String.IsNullOrEmpty(propName))
                    sb.AppendLayerFormatLine(layer, "[null]");
                else
                    sb.AppendLayerFormatLine(layer, "{0}: [null]", propName);
            }
            else
            {
                if (!String.IsNullOrEmpty(propName))
                {
                    string strSize = "";
                    if (collectionInstance.GetType().IsArray)
                    {
                        System.Array ar = collectionInstance as System.Array;
                        int rank = ar.Rank;
                        StringBuilder ars = new StringBuilder();
                        for (int i = 0; i < rank; i++)
                        {
                            if (i != 0)
                                ars.Append("*");
                            ars.Append(ar.GetLength(i));
                        }
                        strSize = String.Format("[size: {0}]", ars.ToString());
                    }

                    sb.AppendLayerFormatLine(layer, "{0}: {1}", propName, strSize);
                }

                foreach (object item in collectionInstance as IEnumerable)
                {
                    if (IsBasicType(item.GetType()) || item == null)
                    {
                        sb.AppendLayerFormatLine(layer + 1, "[{0}]", item == null ? "null" : item.ToString());
                    }
                    else
                    {
                        sb.AppendLayerFormatLine(layer + 1, "[");
                        sb.Append(GeneralToString(item, layer + 2));
                        sb.AppendLayerFormatLine(layer + 1, "]");
                    }
                }
            }
            return sb.ToString();
        }

        private static string NonGenericDictionaryToString(object dicInstance, string propName, int layer)
        {
            StringBuilder sb = new StringBuilder();
            if (dicInstance == null)
            {
                if (String.IsNullOrEmpty(propName))
                    sb.AppendLayerFormatLine(layer, "[null]");
                else
                    sb.AppendLayerFormatLine(layer, "{0}: [null]", propName);
            }
            else
            {
                if (!String.IsNullOrEmpty(propName))
                    sb.AppendLayerFormatLine(layer, "{0}:", propName);

                foreach (object pair in dicInstance as IEnumerable)
                {
                    if (pair == null)
                    {
                        sb.AppendLayerFormatLine(layer + 1, "[null]");
                    }
                    else
                    {
                        sb.AppendLayerFormatLine(layer + 1, "[");

                        object objKey = pair.GetType().GetProperty("Key").GetValue(pair, null);
                        object objValue = pair.GetType().GetProperty("Value").GetValue(pair, null);

                        if (objKey == null || IsBasicType(objKey.GetType()) )
                        {
                            sb.AppendLayerFormatLine(layer + 1, "Key: {0}",
                                objKey == null ? "[null]" : objKey.ToString()
                            );
                        }
                        else
                        {
                            sb.AppendLayerFormatLine(layer + 1, "Key: ");
                            sb.AppendLayerFormatLine(layer + 2, "[");
                            sb.Append(GeneralToString(objKey, layer + 3));
                            sb.AppendLayerFormatLine(layer + 2, "]");
                        }

                        if (objValue == null || IsBasicType(objValue.GetType()))
                        {
                            sb.AppendLayerFormatLine(layer + 1, "Value: {0}",
                                objValue == null ? "[null]" : objValue.ToString()
                            );
                        }
                        else
                        {
                            sb.AppendLayerFormatLine(layer + 1, "Value: ");
                            sb.AppendLayerFormatLine(layer + 2, "[");
                            sb.Append(GeneralToString(objValue, layer + 3));
                            sb.AppendLayerFormatLine(layer + 2, "]");
                        }

                        sb.AppendLayerFormatLine(layer + 1, "]");
                    }
                }
            }

            return sb.ToString();
        }


        private static string DictionaryToString(object dicInstance, string propName, int layer)
        {
            StringBuilder sb = new StringBuilder();
            if (dicInstance == null)
            {
                if(String.IsNullOrEmpty(propName))
                    sb.AppendLayerFormatLine(layer, "[null]");
                else
                    sb.AppendLayerFormatLine(layer, "{0}: [null]", propName);
            }
            else
            {
                if (!String.IsNullOrEmpty(propName))
                    sb.AppendLayerFormatLine(layer, "{0}:", propName);

                Type keyType, valueType;
                IsDictionary(dicInstance.GetType(), out keyType, out valueType);
                if (IsBasicType(keyType) && IsBasicType(valueType))
                {
                    object objKey, objValue;
                    foreach (object pair in dicInstance as IEnumerable)
                    {
                        if (pair == null)
                        {
                            sb.AppendLayerFormatLine(layer + 1, "[null]");
                        }
                        else
                        {
                            objKey = pair.GetType().GetProperty("Key").GetValue(pair, null);
                            objValue = pair.GetType().GetProperty("Value").GetValue(pair, null);
                            sb.AppendLayerFormatLine(layer + 1, "[{0} -> {1}]",
                                objKey == null ? "[null]" : objKey.ToString(),
                                objValue == null ? "[null]" : objValue.ToString()
                            );
                        }
                    }
                }
                else
                {
                    object objKey, objValue;
                    foreach (object pair in dicInstance as IEnumerable)
                    {
                        if (pair == null)
                        {
                            sb.AppendLayerFormatLine(layer + 1, "[null]");
                        }
                        else
                        {
                            sb.AppendLayerFormatLine(layer + 1, "[");

                            objKey = pair.GetType().GetProperty("Key").GetValue(pair, null);
                            objValue = pair.GetType().GetProperty("Value").GetValue(pair, null);

                            if (IsBasicType(keyType) || objKey == null)
                            {
                                sb.AppendLayerFormatLine(layer + 1, "Key: {0}",
                                    objKey == null ? "[null]" : objKey.ToString()
                                );
                            }
                            else
                            {
                                sb.AppendLayerFormatLine(layer + 1, "Key: ");
                                sb.AppendLayerFormatLine(layer + 2, "[");
                                sb.Append(GeneralToString(objKey, layer + 3));
                                sb.AppendLayerFormatLine(layer + 2, "]");

                            }

                            if (IsBasicType(valueType) || objValue == null)
                            {
                                sb.AppendLayerFormatLine(layer + 1, "Value: {0}",
                                    objValue == null ? "[null]" : objValue.ToString()
                                );
                            }
                            else
                            {
                                sb.AppendLayerFormatLine(layer + 1, "Value: ");
                                sb.AppendLayerFormatLine(layer + 2, "[");
                                sb.Append(GeneralToString(objValue, layer + 3));
                                sb.AppendLayerFormatLine(layer + 2, "]");
                            }

                            sb.AppendLayerFormatLine(layer + 1, "]");
                        }
                    }
                }
            }
            return sb.ToString();
        }

        private static string GeneralToString(object o, int layer)
        {
            StringBuilder sb = new StringBuilder();
            if (o == null)
            {
                sb.AppendLayerFormatLine(layer, "[null]");
                //sb.AppendLine("[null]");
            }
            else if (IsBasicType(o.GetType()))
            {
                sb.AppendLayerFormatLine(layer, o.ToString());
            }
            else if (IsDictionary(o.GetType()))
            {
                sb.Append(DictionaryToString(o, null, layer));
            }
            else if (IsCollection(o.GetType()))
            {
                sb.Append(CollectionToString(o, null, layer));
            }
            else if (IsIFormattable(o.GetType()))
            {
                sb.AppendLayerFormatLine(layer, o.ToString());
            }
            else
            {
                Type propType;
                foreach (var prop in o.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public))
                {
                    if (!(prop.CanRead))
                        continue;

                    if(prop.GetIndexParameters().Length > 0) // do not print indexers
                        continue;

                    if (ReflectionUtils.IsTypeEqualOrInheritedFromType(prop.PropertyType, typeof(Delegate)))
                        continue; // do not print delegates

                    propType = prop.PropertyType;
                    if (IsBasicType(propType))
                    {
                        sb.AppendLayerFormatLine(layer, "{0}: {1}", prop.Name, GetBasicPropertyValue(o, prop));
                    }
                    else if (IsDictionary(propType))
                    {
                        sb.Append(DictionaryToString(prop.GetValue(o, null), prop.Name, layer));
                    }
                    else if (IsCollection(propType))
                    {
                        sb.Append(CollectionToString(prop.GetValue(o, null), prop.Name, layer));
                    }
                    else if (IsNonGenericIDictionary(propType))
                    {
                        sb.Append(NonGenericDictionaryToString(prop.GetValue(o, null), prop.Name, layer));
                    }
                    else if (IsNonGenericIEnumerable(propType))
                    {
                        sb.Append(CollectionToString(prop.GetValue(o, null), prop.Name, layer));
                    }
                    else
                    {
                        object propValue = prop.GetValue(o, null);
                        if (propValue == null)
                        {
                            sb.AppendLayerFormatLine(layer, "{0}: [null]", prop.Name);
                        }
                        else
                        {
                            sb.AppendLayerFormatLine(layer, "{0}:", prop.Name);

                            sb.AppendLayerFormatLine(layer, "[");
                            sb.Append(GeneralToString(propValue, layer + 1));
                            sb.AppendLayerFormatLine(layer, "]");
                        }
                    }
                }
            }

            return sb.ToString();
        }

        private static bool IsDictionary(Type type)
        {
            if (type.IsGenericType)
                type = type.GetGenericTypeDefinition();

            if (type == typeof(Dictionary<,>))
                return true;

            return false;
        }

        /// <summary>
        /// Determines whether the specified type is a generic dictionary.
        /// </summary>
        /// <param name="type">The type to check.</param>
        /// <param name="keyType">Type of the key.</param>
        /// <param name="valueType">Type of the value.</param>
        /// <returns>
        /// 	<c>true</c> if the specified type is dictionary; otherwise, <c>false</c>.
        /// </returns>
        private static bool IsDictionary(Type type, out Type keyType, out Type valueType)
        {
            keyType = typeof(object);
            valueType = typeof(object);

            foreach (Type interfaceType in type.GetInterfaces())
            {
                if (interfaceType.IsGenericType &&
                    interfaceType.GetGenericTypeDefinition() == typeof(IDictionary<,>))
                {
                    Type[] genArgs = interfaceType.GetGenericArguments();
                    keyType = genArgs[0];
                    valueType = genArgs[1];
                    return true;
                }

            }

            return false;
        }

        private static bool IsNonGenericIDictionary(Type type)
        {
            if (type == typeof(IDictionary))
                return true;

            foreach (Type interfaceType in type.GetInterfaces())
            {
                if (interfaceType == typeof(IDictionary))
                {
                    return true;
                }

            }

            return false;

        }

        private static bool IsNonGenericIEnumerable(Type type)
        {
            if (type == typeof(IEnumerable))
                return true;

            foreach (Type interfaceType in type.GetInterfaces())
            {
                if (interfaceType == typeof(IEnumerable))
                {
                    return true;
                }
            }

            return false;
        }


        private static bool IsCollection(Type type)
        {
            if (type == typeof(string)) 
                return false;

            if (IsArray(type)) 
                return true;

            if (type.IsGenericType)
                type = type.GetGenericTypeDefinition();

            if (type == typeof(List<>) || type == typeof(HashSet<>) || type == typeof(IEnumerable<>))
                return true;

            Type elemType;
            if(IsIEnumerableExceptArray(type, out elemType))
            {
                return true;
            }

            return false;
        }

        private static bool IsIFormattable(Type type)
        {
            // is IFormattable
            // accept parameterless ToString
            // accept ctor of string
            foreach (Type interfaceType in type.GetInterfaces())
            {
                if (interfaceType == typeof(IFormattable))
                {
                    if (!HasSuitableProperties(type))
                    {
                        if (null != type.GetConstructor(new Type[] { typeof(string) }))
                        {
                            if (null != type.GetMethod("ToString", new Type[0]) &&
                                null != type.GetMethod("ToString", new Type[] { typeof(string) }))
                                return true;
                        }
                    }
                }
            }

            return false;
        }

        private static bool HasSuitableProperties(Type type)
        {
            var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            foreach (var pi in props)
            {
                if (pi.CanRead && pi.CanWrite)
                {
                    var getPi = pi.GetGetMethod(false);
                    var setPi = pi.GetSetMethod(false);
                    if (setPi != null && getPi != null)
                        return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Gets the type of the items of a collection type.
        /// </summary>
        /// <param name="type">The type of the collection.</param>
        /// <returns>the type of the items of a collection type.</returns>
        private static Type GetCollectionItemType(Type type)
        {
            Type itemType = typeof(object);

            if (type.IsInterface && type.GetGenericTypeDefinition() == typeof(IEnumerable<>))
            {
                itemType = type.GetGenericArguments()[0];
            }
            else if (type.IsInterface && type == typeof(IEnumerable))
            {
                itemType = typeof(object);
            }
            else
            {
                foreach (Type interfaceType in type.GetInterfaces())
                {
                    if (interfaceType.IsGenericType &&
                        interfaceType.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                    {
                        itemType = interfaceType.GetGenericArguments()[0];
                    }
                }
            }

            return itemType;
        }



        private static string GetBasicPropertyValue(object o, PropertyInfo prop)
        {
            object value = prop.GetValue(o, null);
            return (value == null) ? "[null]" : value.ToString();
        }

        /// <summary>
        /// Determines whether the specified type is basic type. A basic type is one that can be wholly expressed
        /// as an XML attribute. All primitive data types and type <c>string</c> and <c>DataTime</c> are basic.
        /// </summary>
        /// <param name="t">The type</param>
        private static bool IsBasicType(Type t)
        {
            if (t == typeof(string) || t.IsPrimitive || t.IsEnum || t == typeof(DateTime) || t == typeof(decimal))
                return true;
            else
                return false;
        }

        /// <summary>
        /// Determines whether the specified type is array.
        /// </summary>
        /// <param name="t">The type</param>
        /// <returns>
        /// 	<c>true</c> if the specified type is array; otherwise, <c>false</c>.
        /// </returns>
        private static bool IsArray(Type t)
        {
            return (t.BaseType == typeof(System.Array));
        }

        /// <summary>
        /// Determines whether the specified type has implemented or is an <c>IEnumerable</c> or <c>IEnumerable&lt;&gt;</c>.
        /// This method does not detect Arrays.
        /// </summary>
        /// <param name="type">The type to check.</param>
        /// <param name="seqType">Type of the sequence items.</param>
        /// <returns>
        /// <value><c>true</c> if the specified type is enumerable; otherwise, <c>false</c>.</value>
        /// </returns>
        public static bool IsIEnumerableExceptArray(Type type, out Type seqType)
        {
            seqType = typeof(object);
            if (type == typeof(IEnumerable))
                return true;

            bool isNongenericEnumerable = false;

            if (type.IsInterface && type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IEnumerable<>))
            {
                seqType = type.GetGenericArguments()[0];
                return true;
            }

            foreach (Type interfaceType in type.GetInterfaces())
            {
                if (interfaceType.IsGenericType &&
                    interfaceType.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                {
                    Type[] genArgs = interfaceType.GetGenericArguments();
                    seqType = genArgs[0];
                    return true;
                }
                else if (interfaceType == typeof(IEnumerable))
                {
                    isNongenericEnumerable = true;
                }
            }

            // the second case is a direct reference to IEnumerable
            if (isNongenericEnumerable || type == typeof(IEnumerable))
            {
                seqType = typeof(object);
                return true;
            }

            return false;
        }

    }

    public static class StringBuilderExtensions
    {
        public static StringBuilder AppendLayerFormatLine(this StringBuilder sb, int layer, string format, params object[] args)
        {
            return AppendLayerFormat(sb, layer, format + Environment.NewLine, args);
        }

        public static StringBuilder AppendLayerFormat(this StringBuilder sb, int layer, string format, params object[] args)
        {
            string strToAppend = String.Format(format, args);
            return sb.AppendFormat("{0}{1}", GetLayerPrefix(layer), strToAppend);
        }

        private static string GetLayerPrefix(int layer)
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < layer; ++i)
                sb.Append("   ");

            return sb.ToString();
        }
    }
}
