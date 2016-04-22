using GoodAI.Logging;
using System;
using System.Collections.Generic;

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
                    message.Objects, message.Exception));
            }

            return list;
        }

        private static TWSeverity ToTWSeverity(Severity s)
        {
            return (TWSeverity) s;
        }
    }
}