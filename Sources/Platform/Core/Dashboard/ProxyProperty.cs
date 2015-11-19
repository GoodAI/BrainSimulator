using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Dashboard
{
    public class ProxyProperty
    {
        private object Owner { get; set; }
        private PropertyInfo PropertyInfo { get; set; }

        public ProxyProperty(object owner, PropertyInfo propertyInfo)
        {
            Owner = owner;
            PropertyInfo = propertyInfo;
            Visible = true;
        }

        public virtual string Name { get; set; }

        public bool ReadOnly { get; set; }
        public bool Visible { get; set; }

        public object Value
        {
            get { return PropertyInfo.GetValue(Owner); }
            set { PropertyInfo.SetValue(Owner, value); }
        }

        public string Category { get; set; }
    }
}
