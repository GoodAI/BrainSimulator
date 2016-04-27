using GoodAI.Logging;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Logger
{
    public class TWLog
    {

        public static List<TWLogMessage> GetAllLogMessages()
        {
            var list = new List<TWLogMessage>();
            ILogReader reader = LogReader.Instance;
            

            while(true)
            {
                LogMessage message = reader.GetNextLogMessage();
                if (message == null)
                {
                    break;
                }
                list.Add(new TWLogMessage(ToTWSeverity(message.Severity), message.Template, 
                    message.Objects, message.Exception, message.Time));
            }

            return list;
        }

        public static List<TWLogMessage> GetVerboseLogMessages()
        {
            return GetAllLogMessages();
        }

        public static List<TWLogMessage> GetDebugLogMessages()
        {
            return GetVerboseLogMessages().Where(x => x.Severity != TWSeverity.Verbose).ToList();
        }

        public static List<TWLogMessage> GetInfoLogMessages()
        {
            return GetDebugLogMessages().Where(x => x.Severity != TWSeverity.Debug).ToList();
        }

        public static List<TWLogMessage> GetWarnLogMessages()
        {
            return GetInfoLogMessages().Where(x => x.Severity != TWSeverity.Info).ToList();
        }

        public static List<TWLogMessage> GetErrorLogMessages()
        {
            return GetWarnLogMessages().Where(x => x.Severity != TWSeverity.Warn).ToList();
        }

        private static TWSeverity ToTWSeverity(Severity s)
        {
            return (TWSeverity) s;
        }
    }
}