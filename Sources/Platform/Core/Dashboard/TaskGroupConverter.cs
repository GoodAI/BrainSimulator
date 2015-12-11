using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Task;

namespace GoodAI.Core.Dashboard
{

    class TaskGroupConverter : TypeConverter
    {
        private readonly List<string> m_taskNames = new List<string>();

        public TaskGroupConverter(IEnumerable<string> taskNames)
        {
            m_taskNames.Add("");
            m_taskNames.AddRange(taskNames);
        }

        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            if (sourceType == typeof (string))
                return true;

            return base.CanConvertFrom(context, sourceType);
        }

        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            var strValue = value as string;
            if (strValue != null && m_taskNames.Contains(strValue))
                return strValue;

            return base.ConvertFrom(context, culture, value);
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            return new StandardValuesCollection(m_taskNames.ToArray());
        }

        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context)
        {
            return base.GetStandardValuesExclusive(context);
        }

        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }
    }
}
