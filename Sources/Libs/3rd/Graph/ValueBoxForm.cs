using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Graph
{
    public partial class ValueBoxForm : Form
    {
        public string InputText
        {
            get { return valueTextBox.Text; }
            set { valueTextBox.Text = value; }
        }

        public Type InputType
        {
            get { return SupportedTypes[typeComboBox.SelectedIndex]; }
            set { typeComboBox.SelectedItem = value.Name; }
        }

        public bool EnableTypeComboBox
        {
            get { return typeComboBox.Enabled; }
            set { typeComboBox.Enabled = value; }
        }

        public Type[] SupportedTypes { get; private set; }
        public void SetSupportedTypes(Type[] types)
        {
            SupportedTypes = types;
            foreach (var type in types)
            {
                typeComboBox.Items.Add(type.Name);
            }
        }

        public ValueBoxForm()
        {
            InitializeComponent();
        }
    }
}
