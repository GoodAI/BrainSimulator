using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.Forms
{
    public partial class LicensesForm : Form
    {
        public LicensesForm()
        {
            InitializeComponent();
        }

        private void LicensesForm_Load(object sender, EventArgs e)
        {
            this.Text = AssemblyTitle + " Licensing Information";

            try
            {
                LoadLicensesList();
            }
            catch (Exception exc)
            {
                MessageBox.Show("Error loading licenses.\n\n" + exc.Message);
            }
        }

        private void LoadLicensesList()
        {
            var licensesDirInfo = new DirectoryInfo(Path.Combine(MyResources.GetEntryAssemblyPath(), @"licenses"));

            foreach (var file in licensesDirInfo.GetFiles("*.txt"))
            {
                licenseList.Items.Add(new LicenseItem(file.FullName));
            }
        }

        // TODO(Premek): resolve duplicity with AboutDialog
        public string AssemblyTitle
        {
            get
            {
                object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyTitleAttribute), false);
                if (attributes.Length > 0)
                {
                    AssemblyTitleAttribute titleAttribute = (AssemblyTitleAttribute)attributes[0];
                    if (titleAttribute.Title != "")
                    {
                        return titleAttribute.Title;
                    }
                }
                return System.IO.Path.GetFileNameWithoutExtension(Assembly.GetExecutingAssembly().CodeBase);
            }
        }

        private class LicenseItem
        {
            public string FileName { get; private set; }
            public string Name { get; private set; }

            public LicenseItem(string fileFullName)
            {
                FileName = fileFullName;
                Name = Path.GetFileNameWithoutExtension(fileFullName);
            }

            public override string ToString()
            {
                return Name;
            }
        }

        private void licenseList_SelectedIndexChanged(object sender, EventArgs e)
        {
            var item = (LicenseItem)licenseList.SelectedItem;

            try
            {
                licenseText.Text = File.ReadAllText(item.FileName);
            }
            catch (Exception exc)
            {
                licenseText.Text = "Error reading file " + item.FileName + "\n\n" + exc.Message;
            }
        }
    }
}
