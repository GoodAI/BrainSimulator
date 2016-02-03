using GoodAI.Core.Nodes;
using GoodAI.Modules.School.Common;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using YAXLib;

namespace GoodAI.Modules.School.Worlds
{
    public interface IWorldAdapter
    {
        void InitAdapterMemory(SchoolWorld schoolWorld);
        void InitWorldInputs(int nGPU, SchoolWorld schoolWorld);
        void MapWorldInputs(SchoolWorld schoolWorld);
        void InitWorldOutputs(int nGPU, SchoolWorld schoolWorld);
        void MapWorldOutputs(SchoolWorld schoolWorld);
        void ClearWorld();
        void SetHint(TSHintAttribute attr, float value);
    }

    public static class IWorldAdaptersList
    {
        public static Type[] Types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(s => s.GetTypes())
                .Where(p => typeof(IWorldAdapter).IsAssignableFrom(p) && p.IsClass && !p.IsAbstract).ToArray();
    }
    
    public class IWorldAdapterConverter: TypeConverter
    {
        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            if (sourceType == typeof(string))
            {
                return true;
            }
            return base.CanConvertFrom(context, sourceType);
        }
        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value is string)
            {
                return (IWorldAdapter)(Activator.CreateInstance(Type.GetType((string)value))); 
            }
            if (value == null)
            {
                return "";
            }
            return base.ConvertFrom(context, culture, value);
        }
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }
        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            //Call the abstract GetValues function here.
            return new StandardValuesCollection(GetValues());
        }
        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context)
        {
            return true;
        }

        protected Type[] GetValues() 
        {
            return IWorldAdaptersList.Types;
        }
    }

    public class WorldAdapterSerializer : ICustomSerializer<IWorldAdapter>
    {
        public IWorldAdapter DeserializeFromAttribute(XAttribute attrib)
        {
            return ConvertToIWorldAdapter(attrib.Value);
        }

        public IWorldAdapter DeserializeFromElement(XElement element)
        {
            return ConvertToIWorldAdapter(element.Value);
        }

        public IWorldAdapter DeserializeFromValue(string value)
        {
            return ConvertToIWorldAdapter(value);
        }

        public void SerializeToAttribute(IWorldAdapter objectToSerialize, XAttribute attrToFill)
        {
            if (objectToSerialize != null)
                attrToFill.Value = objectToSerialize.GetType().ToString();
        }

        public void SerializeToElement(IWorldAdapter objectToSerialize, XElement elemToFill)
        {
            if (objectToSerialize != null)
                elemToFill.Value = objectToSerialize.GetType().ToString();
        }

        public string SerializeToValue(IWorldAdapter objectToSerialize)
        {
            if (objectToSerialize != null)
                return objectToSerialize.GetType().ToString();
            else
                return String.Empty;
        }
 
        private static IWorldAdapter ConvertToIWorldAdapter(string attributeString)
        {
            return (IWorldAdapter)(Activator.CreateInstance(Type.GetType(attributeString)));
        }
    }

}
