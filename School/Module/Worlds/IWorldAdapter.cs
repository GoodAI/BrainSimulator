using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Modules.School.Common;
using System;
using System.ComponentModel;
using System.Globalization;
using System.Linq;

namespace GoodAI.Modules.School.Worlds
{
    public interface IWorldAdapter
    {
        MyWorkingNode World { get; }
        SchoolWorld School { get; set; }

        MyTask GetWorldRenderTask();
        void InitAdapterMemory();
        void InitWorldInputs(int nGPU);
        void MapWorldInputs();
        void InitWorldOutputs(int nGPU);
        void MapWorldOutputs();
        void ClearWorld();
        void UpdateMemoryBlocks();
        void SetHint(TSHintAttribute attr, float value);
    }

    public static class IWorldAdaptersList
    {
        public static Type[] Types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(s => s.GetTypes())
                .Where(p => typeof(IWorldAdapter)
                .IsAssignableFrom(p) && p.IsClass && !p.IsAbstract)
                .ToArray();
    }

    public class IWorldAdapterConverter : TypeConverter
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
}
