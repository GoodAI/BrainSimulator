using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Logging
{
    public static class LoggerExtensions
    {
        public static void Error(this ILog log, string template, params object[] objects)
        {
            log.Add(Severity.Error, template, objects);
        }

        public static void Error(this ILog log, Exception ex, string template, params object[] objects)
        {
            log.Add(Severity.Error, ex, template, objects);
        }

        public static void Warn(this ILog log, string template, params object[] objects)
        {
            log.Add(Severity.Warn, template, objects); 
        }

        public static void Warn(this ILog log, Exception ex, string template, params object[] objects)
        {
            log.Add(Severity.Warn, ex, template, objects);
        }

        public static void Info(this ILog log, string template, params object[] objects)
        {
            log.Add(Severity.Info, template, objects);
        }

        public static void Debug(this ILog log, string template, params object[] objects)
        {
            log.Add(Severity.Debug, template, objects);
        }

        public static void Debug(this ILog log, Exception ex, string template, params object[] objects)
        {
            log.Add(Severity.Debug, ex, template, objects);
        }
    }
}
