using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.BrainSimulator.Properties;
using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.UserSettings
{
    internal static class AppSettings
    {
        public static MyLogLevel GetInitialLogLevel()
        {
            try
            {
                return (MyLogLevel) Settings.Default.LogLevel;
            }
            catch (Exception e)  // ConfigurationErrorsException
            {
                MyLog.ERROR.WriteLine("Error reading log level from configuration (Resetting user config!): "
                    + e.Message);

                ResetUserConfiguration();
                
                return MyLogLevel.INFO;
            }
        }

        private static void ResetUserConfiguration()
        {
            try
            {
                Settings.Default.Reset();
            }
            catch (Exception ex)
            {
                MyLog.ERROR.WriteLine("Error resetting user configuration: " + ex.Message);
            }
        }

        internal static void SaveSettings(Action<Settings> action)
        {
            Settings settings = Settings.Default;

            action(settings);

            // The settings are only saved if no handlers threw anything.
            // This is because Settings is INotifyPropertyChanged and the app might crash as a result of a setting, 
            // and if it gets saved, it might cause the app to crash at startup next time.
            settings.Save();
        }
    }
}
