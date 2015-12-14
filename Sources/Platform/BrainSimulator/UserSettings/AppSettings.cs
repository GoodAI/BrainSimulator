using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.BrainSimulator.Properties;
using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.UserSettings
{
    public static class AppSettings
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
    }
}
