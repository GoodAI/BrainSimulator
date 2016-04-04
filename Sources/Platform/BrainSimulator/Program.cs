using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using GoodAI.BrainSimulator.Forms;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;

namespace GoodAI.BrainSimulator
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            UnhandledExceptionCatcher.RegisterHandlers();

            ConfigureTypeMap();

            Environment.CurrentDirectory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            Application.Run(new MainForm());
        }

        private static void ConfigureTypeMap()
        {
            TypeMap.InitializeConfiguration<CoreContainerConfiguration>();
            TypeMap.Verify();
        }
    }
}
    