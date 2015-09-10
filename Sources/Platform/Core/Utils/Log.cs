using System;
using System.IO;
using System.Text;

namespace GoodAI.Core.Utils
{
    public enum MyLogLevel 
    {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };

    public interface MyLogWriter
    {
        void WriteLine(MyLogLevel level, string message);
        void Write(MyLogLevel level, string message);
        void Write(MyLogLevel level, char message);
        void FlushCache();
    }

    public class MyLog : TextWriter
    {
        public static MyLog DEBUG { get; private set; }
        public static MyLog INFO { get; private set; }
        public static MyLog WARNING { get; private set; }
        public static MyLog ERROR { get; private set; }

        public static MyLogLevel Level { get; set; }
        public static MyLogWriter Writer { get; set; }

        private MyLogLevel m_level;

        static MyLog()
        {
            DEBUG = new MyLog(MyLogLevel.DEBUG);
            INFO = new MyLog(MyLogLevel.INFO);
            WARNING = new MyLog(MyLogLevel.WARNING);
            ERROR = new MyLog(MyLogLevel.ERROR);

            MyLog.Writer = new MyDefaultLogWriter();
            MyLog.Level = MyLogLevel.INFO;
        }        
       
        public static void GrabConsole() 
        {
            Console.SetOut(MyLog.INFO);
        }        

        private MyLog(MyLogLevel level)
        {
            m_level = level;
        }

        public override void WriteLine(string message)
        {
            if (m_level >= MyLog.Level)
            {
                MyLog.Writer.WriteLine(m_level, message);
            }
        }

        public override void Write(string value)
        {
            if (m_level >= MyLog.Level)
            {
                MyLog.Writer.Write(m_level, value);
            }
        }

        public override void Write(char value)
        {
            if (m_level >= MyLog.Level)
            {
                MyLog.Writer.Write(m_level, value);
            }
        }

        public override Encoding Encoding
        {
            get { return System.Text.Encoding.UTF8; }
        }

        public class MyDefaultLogWriter : MyLogWriter
        {
            private StreamWriter m_stdOut;

            public MyDefaultLogWriter()
            {
                m_stdOut = new StreamWriter(Console.OpenStandardOutput());
            }

            public void Write(MyLogLevel level, char value)
            {
                ConsoleColor tmp = Console.ForegroundColor;
                Console.ForegroundColor = GetConsoleColor(level);
                m_stdOut.Write(value);
                m_stdOut.Flush();
                Console.ForegroundColor = tmp;
            }

            public void Write(MyLogLevel level, string value)
            {
                ConsoleColor tmp = Console.ForegroundColor;
                Console.ForegroundColor = GetConsoleColor(level);
                m_stdOut.Write(value);
                m_stdOut.Flush();
                Console.ForegroundColor = tmp;
            }

            public void WriteLine(MyLogLevel level, string value)
            {
                ConsoleColor tmp = Console.ForegroundColor;
                Console.ForegroundColor = GetConsoleColor(level);
                m_stdOut.WriteLine("[" + level + "] " + value);
                m_stdOut.Flush();
                Console.ForegroundColor = tmp;
            }

            public void FlushCache()
            {

            }

            private static ConsoleColor GetConsoleColor(MyLogLevel level)
            {
                switch (level)
                {
                    case MyLogLevel.DEBUG: return ConsoleColor.Gray;
                    case MyLogLevel.INFO: return ConsoleColor.White;
                    case MyLogLevel.WARNING: return ConsoleColor.Yellow;
                    case MyLogLevel.ERROR: return ConsoleColor.Red;
                }
                return ConsoleColor.White;
            }
        }
    }
}
