using System;

namespace GoodAI.Core.Utils
{
    [AttributeUsage(AttributeTargets.Class)]
    public class MyTaskInfoAttribute : Attribute
    {
        public bool OneShot { get; set; }
        public int Order { get; set; }
        public bool Disabled { get; set; }
        public bool DesignTime { get; set; }

        public MyTaskInfoAttribute()
        {
            OneShot = false;
            Order = 0;
            Disabled = false;
            DesignTime = false;
        }
    }

    [AttributeUsage(AttributeTargets.Property)]
    public class MyTaskGroupAttribute : Attribute
    {
        public MyTaskGroupAttribute(string name)
        {
            Name = name;            
        }

        public string Name { get; private set; }        
    }

    [AttributeUsage(AttributeTargets.Class)]
    public class MyNodeInfoAttribute : Attribute
    {
        public bool FixedOutput { get; set; }     

        public MyNodeInfoAttribute()
        {
            FixedOutput = false;            
        }
    }

    [AttributeUsage(AttributeTargets.Class)]
    public class MyObsoleteAttribute : Attribute
    {
        public Type ReplacedBy { get; set; }
    }

    [AttributeUsage(AttributeTargets.Property)]
    public class MyInputBlockAttribute : Attribute
    {
        public int Order { get; internal set; }

        public MyInputBlockAttribute()
        {
            Order = -1;            
        }

        public MyInputBlockAttribute(int order)
        {
            Order = order;
        }
    }

    [AttributeUsage(AttributeTargets.Property)]
    public class MyOutputBlockAttribute : MyPersistableAttribute
    {
        public int Order { get; internal set; }

        public MyOutputBlockAttribute()
        {
            Order = -1;            
        }

        public MyOutputBlockAttribute(int order)
        {
            Order = order;
        }
    }

    [AttributeUsage(AttributeTargets.Property)]
    public class MyPersistableAttribute : Attribute { }    

    [AttributeUsage(AttributeTargets.Property)]
    public class MyUnmanagedAttribute : Attribute { }    

    [AttributeUsage(AttributeTargets.Property)]
    public class MyBrowsableAttribute : Attribute 
    {
        public bool Browsable { get; protected set; }

        public MyBrowsableAttribute(bool browsable = true)
        {
            Browsable = browsable;
        }
    }

    [AttributeUsage(AttributeTargets.Property)]
    public class DynamicBlockAttribute : Attribute { }
}
