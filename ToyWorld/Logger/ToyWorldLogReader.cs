using System;
using System.Collections.Generic;

namespace Logger
{
    public class ToyWorldLogReader : AbstractLog
    {
        private static ToyWorldLogReader m_toyWorldLogReader;

        public List<Tuple<string, object[], Exception>> DeqeueAll()
        {
            var list = new List<Tuple<string, object[], Exception>>();

            while(true){
                LogMessage message;
                Queue.TryDequeue(out message);
                if (message == null)
                {
                    break;
                }
                var messageTuple = new Tuple<string, object[], Exception>(
                    "<" + SeverityNames[(int) message.Severity]+"> " + message.Template,
                    message.Objects,
                    message.Exception
                    );
                list.Add(messageTuple);
            }

            return list;
        }

        public static ToyWorldLogReader Instance
        {
            get { return m_toyWorldLogReader ?? (m_toyWorldLogReader = new ToyWorldLogReader()); }
        }
    }
}