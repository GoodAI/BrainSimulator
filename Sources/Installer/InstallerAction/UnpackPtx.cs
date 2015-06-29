using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Configuration.Install;
using System.Linq;
using System.IO;
using System.IO.Compression;
using System.Security.Permissions;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace InstallerAction
{
    [RunInstaller(true)]
    public partial class UnpackPtx : System.Configuration.Install.Installer
    {
        public UnpackPtx()
        {
            InitializeComponent();
        }

        [SecurityPermission(SecurityAction.Demand)]
        public override void Install(IDictionary stateSaver)
        {
            base.Install(stateSaver);

            try
            {
                ExtractPtxZip(this.GetTargetDir(), failOnError: true);
            }
            catch (InstallException e)
            {
                throw e;  // forward exception with more specific description
            }
            catch (Exception e)
            {
                throw new InstallException("Unable to unpack CUDA kernels.", e);
            }

            try
            {
                VisitMouduleSubdirs(ExtractPtxZip);
            }
            catch { }  // notable errors are reported
        }

        [SecurityPermission(SecurityAction.Demand)]
        public override void Commit(IDictionary savedState)
        {
            base.Commit(savedState);
        }

        [SecurityPermission(SecurityAction.Demand)]
        public override void Rollback(IDictionary savedState)
        {
            base.Rollback(savedState);
            UninstallOrRollback(showError: false);
        }

        [SecurityPermission(SecurityAction.Demand)]
        public override void Uninstall(IDictionary savedState)
        {
            base.Uninstall(savedState);
            UninstallOrRollback(showError: true);
        }

        private void UninstallOrRollback(bool showError)
        {
            try
            {
                DeleteDir(Path.Combine(this.GetTargetDir(), @"ptx"), showError);

                VisitMouduleSubdirs(DeletePtxSubdir);
            }
            catch (Exception e)
            {
                if (showError)
                    MessageBox.Show("Unable to uninstall some items." + e.Message);
            }
        }

        private void VisitMouduleSubdirs(Action<string> action)
        {
            foreach(var dir in Directory.GetDirectories(Path.Combine(this.GetTargetDir(), @"modules")))
            {
                action(dir);
            }
        }

        private void ExtractPtxZip(string dir)
        {
            ExtractPtxZip(dir, failOnError: false);
        }

        private void ExtractPtxZip(string dir, bool failOnError)
        {
            try
            {
                string ptxZipFileName = Path.Combine(dir, @"ptx.zip");
                if (!File.Exists(ptxZipFileName))
                {
                    if (failOnError)
                        throw new InstallException("Missing CUDA kernels pack in '" + dir + "'");
                    else
                        return;  // no zip archive, and we're OK with that
                }

                // don't fail in case some junk is left by previous installation
                // (we already checked ptx.zip is there to prevent deleting ptx dir when there's nothing to extract)
                DeleteDir(Path.Combine(dir, @"ptx"));
                
                ZipFile.ExtractToDirectory(ptxZipFileName, dir);

                try
                {
                    File.Delete(ptxZipFileName);
                }
                catch { }  // We don't care much about a faluire, the action is not critical
            }
            catch (Exception e)
            {
                if (failOnError)
                    throw new InstallException("Unable to unpack CUDA kernels in " + dir, e);
                else
                    // the zip archive was there, but we were unable to unpack it (I guess it is worth reporting)
                    MessageBox.Show("Unable to unpack CUDA kernels in '" + dir + "'\n" + e.Message); 
            }
        }

        private void DeletePtxSubdir(string dir)
        {
            DeleteDir(Path.Combine(dir, @"ptx"));
        }

        private void DeleteDir(string dir)
        {
            DeleteDir(dir, showError: false);
        }

        private void DeleteDir(string dir, bool showError)
        {
            try
            {
                if (Directory.Exists(dir))  // if it does not exist, we are already done
                    Directory.Delete(dir, recursive: true);
            }
            catch (Exception e)
            {
                if (showError)
                    MessageBox.Show("Unable to delete directory '" + dir + "'." + e.Message);
            }
        }

        private string GetTargetDir()
        {
            string targetDir = this.Context.Parameters["targetdir"];
            if (string.IsNullOrEmpty(targetDir))
                throw new ArgumentException("Missing argument 'targetdir' for a custom installer action.");

            return targetDir;
        }
    }
}
