using System;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms;

namespace GoodAI.School.GUI
{
    public partial class LabelDescriptionControl : UserControl
    {
        private Type m_type;
        public Type Type
        {
            get { return m_type; }
            set
            {
                m_type = value;
                if (m_type == null)
                    return;
                DescriptionAttribute descriptionAtt = m_type.GetCustomAttributes(typeof(DescriptionAttribute), true).FirstOrDefault() as DescriptionAttribute;
                if (descriptionAtt != null)
                    label1.Text = descriptionAtt.Description;
            }
        }

        public LabelDescriptionControl()
        {
            InitializeComponent();
        }
    }
}
