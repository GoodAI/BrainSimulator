using GoodAI.BrainSimulator.Utils;
using System;
using System.Reflection;
using System.Windows.Forms;

namespace GoodAI.BrainSimulator.Forms
{
    partial class AboutDialog : Form
    {
        public AboutDialog()
        {
            InitializeComponent();
            this.Text = String.Format("About {0}", AssemblyTitle);
            this.labelProductName.Text = AssemblyProduct;
            this.labelVersion.Text = String.Format("Version {0}", AssemblyVersion);
            this.labelCopyright.Text = AssemblyCopyright;
            this.labelCompanyName.Text = AssemblyCompany;
            this.textBoxDescription.Text = AssemblyDescription;
        }

        #region Assembly Attribute Accessors

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

        public string AssemblyVersion
        {
            get
            {
                return Assembly.GetExecutingAssembly().GetName().Version.ToString();
            }
        }

        public string AssemblyDescription
        {
            get
            {
                //object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyDescriptionAttribute), false);
                //if (attributes.Length == 0)
                //{
                //    return "";
                //}
                //return ((AssemblyDescriptionAttribute)attributes[0]).Description;

                return
                    "Brain Simulator is a collaborative platform for researchers, developers and high-tech companies to prototype and simulate artificial brain architecture, share knowledge, and exchange feedback." +                    
                    "\r\n\r\nThe platform is designed to simplify collaboration, testing, and the implementation of new theories, and to easily visualize experiments and data. No mathematical or programming background is required to experiment with Brain Simulator modules." +
                    "\r\n\r\nPlease keep in mind that Brain Simulator is still in the PROTOTYPE STAGE OF DEVELOPMENT. GoodAI will continuously improve the platform based on its own research advancement and user feedback.";
            }
        }

        public string AssemblyProduct
        {
            get
            {
                object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyProductAttribute), false);
                if (attributes.Length == 0)
                {
                    return "";
                }
                return ((AssemblyProductAttribute)attributes[0]).Product;
            }
        }

        public string AssemblyCopyright
        {
            get
            {
                object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyCopyrightAttribute), false);
                if (attributes.Length == 0)
                {
                    return "";
                }
                return ((AssemblyCopyrightAttribute)attributes[0]).Copyright;
            }
        }

        public string AssemblyCompany
        {
            get
            {
                object[] attributes = Assembly.GetExecutingAssembly().GetCustomAttributes(typeof(AssemblyCompanyAttribute), false);
                if (attributes.Length == 0)
                {
                    return "";
                }
                return ((AssemblyCompanyAttribute)attributes[0]).Company;
            }
        }
        #endregion

        private void licencesButton_Click(object sender, EventArgs e)
        {
            var licensesForm = new LicensesForm();
            licensesForm.Show();
        }

        private void labelCompanyLink_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            try
            {
                MyDocProvider.Navigate("http://www.goodai.com");
            }
            catch (Exception exc)
            {
                MessageBox.Show("Could not open link. " + exc.Message, ":-(", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }
    }
}
