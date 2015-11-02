using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using Serilog;
using Serilog.Events;
using Serilog.Exceptions;

namespace GoodAI.Logging
{
    public sealed class Logger
    {
        public static Logger Instance { get; private set; }

#if DEBUG
        public const string DefaultOutputTemplate = "{Timestamp:yyyy-MM-dd HH:mm:ss} <{SourceContext:l}> [{Level}] ({ThreadId}): {Message}{NewLine}{Exception}";
        public const LogEventLevel DefaultDebugLevel = LogEventLevel.Debug;
#else
        public const string DefaultOutputTemplate = "{Timestamp:yyyy-MM-dd HH:mm:ss} [{Level}]: {Message}{NewLine}{Exception}";
        public const LogEventLevel DefaultDebugLevel = LogEventLevel.Information;
#endif

        static Logger()
        {
            // Serilog diagnostic output. Serilog won't write its errors into the user-space sinks.
            Serilog.Debugging.SelfLog.Out = Console.Out;

            Instance = new Logger(DefaultConfiguration().CreateLogger());
        }

        private static LoggerConfiguration DefaultConfiguration()
        {
            return new LoggerConfiguration()
                .MinimumLevel.Is(DefaultDebugLevel)
                .Enrich.WithThreadId()
                .Enrich.With(new ExceptionEnricher(new ExceptionDestructurer(), new AggregateExceptionDestructurer()))
                .WriteTo.ColoredConsole(outputTemplate: DefaultOutputTemplate);
        }

        public static void Setup(Func<LoggerConfiguration, LoggerConfiguration> configurationAction)
        {
            Instance = new Logger(configurationAction(DefaultConfiguration()).CreateLogger());
        }

        public ILogger SerilogLogger { get; private set; }

        private Logger(ILogger serilogLogger)
        {
            SerilogLogger = serilogLogger;
        }

        #region Delegates

        public void Debug(string template, params object[] parameters)
        {
            SerilogLogger.Debug(template, parameters);
        }

        public void Debug(Exception exception, string template, params object[] parameters)
        {
            SerilogLogger.Debug(exception, template, parameters);
        }

        public void Information(string template, params object[] parameters)
        {
            SerilogLogger.Information(template, parameters);
        }

        public void Information(Exception exception, string template, params object[] parameters)
        {
            SerilogLogger.Information(exception, template, parameters);
        }

        public void Warning(string template, params object[] parameters)
        {
            SerilogLogger.Warning(template, parameters);
        }

        public void Warning(Exception exception, string template, params object[] parameters)
        {
            SerilogLogger.Warning(exception, template, parameters);
        }

        public void Error(string template, params object[] parameters)
        {
            SerilogLogger.Error(template, parameters);
        }

        public void Error(Exception exception, string template, params object[] parameters)
        {
            SerilogLogger.Error(exception, template, parameters);
        }

        /// <summary>
        /// Adds the Type name in the logged event. As opposed to Serilog's ForContext, this doesn't include
        /// the Type's napespace.
        /// </summary>
        /// <typeparam name="TContextType">The type that should be included in the event.</typeparam>
        /// <returns>New Logger instance</returns>
        public Logger ForContext<TContextType>()
        {
            return new Logger(SerilogLogger.ForContext("SourceContext", typeof(TContextType).Name));
        }

        #endregion
    }
}