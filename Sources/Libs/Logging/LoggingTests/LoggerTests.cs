using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Logging;
using Xunit;
using Serilog;
using Serilog.Core;
using Serilog.Events;
using Serilog.Exceptions;
using Xunit.Sdk;

namespace LoggingTests
{
    public class LoggerTests
    {
        class TestSink : ILogEventSink
        {
            public List<LogEvent> Events { get; private set; }

            public TestSink()
            {
                Events = new List<LogEvent>();
            }

            public void Emit(LogEvent logEvent)
            {
                Events.Add(logEvent);
            }
        }

        //[Fact(Skip="Manual test")]
        [Fact]
        public void ManualTest()
        {
            var infoSink = new TestSink();
            var debugSink = new TestSink();
            var writer = new StringWriter();

            Logger.Setup(configuration =>
            {
                configuration
                    .WriteTo.Sink(debugSink)
                    .WriteTo.Sink(infoSink, restrictedToMinimumLevel: LogEventLevel.Information)
                    .WriteTo.TextWriter(writer, outputTemplate: Logger.DefaultConsoleTemplate);

                return configuration;
            });

            Logger.Instance.Debug("debug 1");
            Logger.Instance.Debug("debug 2: {Message}", "foo");
            Logger.Instance.Information("info 1");
            Logger.Instance.Information("info 2: {AnotherMessage}", "bar");

            var logger = Logger.Instance.ForContext<LoggerTests>();
            logger.Debug("debug 3");
            logger.Information("info 3");

            try
            {
                ThrowSimpleException();
            }
            catch (Exception ex)
            {
                logger.Error(ex, "{Component} failed", "Something");
            }

            try
            {
                ThrowAggregateException();
            }
            catch (Exception ex)
            {
                logger.Error(ex, "Multiple failures in {Component}", "SomethingElse");
            }

            try
            {
                ThrowInnerException();
            }
            catch (Exception ex)
            {
                logger.Error(ex, "{Component} failed because of an inner exception", "SomethingWeird");
            }

            var debugResult = debugSink.ToString();
            var infoResult = infoSink.ToString();
            var writerResult = writer.ToString();
        }

        private void ThrowSimpleException()
        {
            throw new Exception("aaa");
        }

        private void ThrowAggregateException()
        {
            var results = new List<Exception>();
            for (int i = 0; i < 3; i++)
            {
                try
                {
                    ThrowSimpleException();
                }
                catch (Exception ex)
                {
                    results.Add(ex);
                }
            }
            throw new AggregateException("ccc", results);
        }

        private void ThrowInnerException()
        {
            try
            {
                ThrowSimpleException();
            }
            catch (Exception ex)
            {
                throw new Exception("bbb", ex);
            }
        }
    }
}
