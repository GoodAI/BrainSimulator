using GoodAI.BrainSimulator.Utils;
using GoodAI.Core.Utils;
using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

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

            if (licenseList.Items.Count > 0)
                licenseList.SelectedIndex = 0;
        }

        private void licenseList_SelectedIndexChanged(object sender, EventArgs e)
        {
            var item = (LicenseItem)licenseList.SelectedItem;

            try
            {
                if (Path.GetExtension(item.FileName).Equals(".rtf"))
                {
                    licenseText.LoadFile(item.FileName);
                }
                else
                {
                    ShowPlainTextLicense(File.ReadAllText(item.FileName));
                }
            }
            catch (Exception exc)
            {
                licenseText.Text = "Error reading file " + item.FileName + "\n\n" + exc.Message;
            }
        }

        private void ShowPlainTextLicense(string text)
        {
            // Always reset font before loading a text file.
            licenseText.Text = "";
            licenseText.Font = new Font("Georgia", 10.5f); // Same as we use in our license RTF file.

            // Show the text.
            licenseText.Text = text;
            
            // Add some hackish margins.
            const int margin = 5;
            licenseText.SelectAll();
            licenseText.SelectionIndent += margin;
            licenseText.SelectionRightIndent += margin;
            licenseText.DeselectAll();
        }

        private void LoadLicensesList()
        {
            // first add our own license
            var eulaFile = Path.Combine(MyResources.GetEntryAssemblyPath(), @"EULA.rtf");

            if (File.Exists(eulaFile))
                licenseList.Items.Add(new LicenseItem(eulaFile, "GoodAI Brain Simulator"));

            // add licenses for 3rd party libs
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

            public LicenseItem(string fileFullName, string displayName)
            {
                FileName = fileFullName;
                Name = displayName;
            }

            public override string ToString()
            {
                return Name;
            }
        }

        private void licenseText_LinkClicked(object sender, LinkClickedEventArgs e)
        {
            try
            {
                MyDocProvider.Navigate(e.LinkText);
            }
            catch (Exception exc)
            {
                MessageBox.Show("Could not open link. " + exc.Message, ":-(", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }
    }
}
