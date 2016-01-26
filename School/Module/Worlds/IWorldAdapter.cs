using GoodAI.Core.Nodes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;

namespace GoodAI.Modules.School.Worlds
{
    public interface IWorldAdapter
    {
        MyWorld World { get; }
    }

    public class AAWorld: MyWorld, IWorldAdapter
    {
        public MyWorld World {
            get
            {
                return this;
            }
        }

        public override void UpdateMemoryBlocks()
        {
        }
    }

    public class BBWorld : MyWorld, IWorldAdapter
    {
        public MyWorld World
        {
            get
            {
                return this;
            }
        }

        public override void UpdateMemoryBlocks()
        {
        }
    }

    public static class IWorldAdaprersList
    {
        public static Type[] Types = AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(s => s.GetTypes())
                .Where(p => typeof(IWorldAdapter).IsAssignableFrom(p) && p.IsClass).ToArray();
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
            return IWorldAdaprersList.Types;
        }
    }
}
