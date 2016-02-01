using GoodAI.Core.Nodes;
using GoodAI.Modules.School.Common;
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
        void InitAdapterMemory(SchoolWorld schoolWorld);
        void MapWorlds(SchoolWorld schoolWorld);
        void ClearWorld();
        void SetHint(string attr, float value);
    }

    /*
    public class SchoolTetris : MyTetrisWorld, IWorldAdapter
    { 
    
        setInput(visual, tetrisvisual);

    }
     */

    public class AAWorld: MyWorld, IWorldAdapter
    {
 
        public override void UpdateMemoryBlocks()
        {

        }

        public void ClearWorld() 
        {
        
        }

        public void InitAdapterMemory(SchoolWorld schoolWorld)
        { 
        
        }

        public void SetHint(string attr, float value)
        {

        }

        public void MapWorlds(SchoolWorld schoolWorld)
        { 
            //schoolWorld.Visual.CopyFromMemoryBlock(World.Whatever)
        }
    }

    public class BBWorld : MyWorld, IWorldAdapter
    {

        public override void UpdateMemoryBlocks()
        {

        }

        public void ClearWorld()
        {

        }

        public void SetHint(string attr, float value)
        {

        }

        public void InitAdapterMemory(SchoolWorld schoolWorld)
        {

        }

        public void MapWorlds(SchoolWorld schoolWorld)
        {
            //schoolWorld.Visual.CopyFromMemoryBlock(World.Whatever)
        }
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
}
