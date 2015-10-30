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
        private class TestSink : ILogEventSink
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

        [Fact]
        public void ManualTest()
        {
            var infoSink = new TestSink();
            var debugSink = new TestSink();

            Logger.Setup(configuration =>
            {
                configuration
                    .WriteTo.Sink(debugSink)
                    .WriteTo.Sink(infoSink, restrictedToMinimumLevel: LogEventLevel.Information);

                return configuration;
            });

            Logger.Instance.Debug("debug 1");
            Logger.Instance.Debug("debug 2: {Message}", "foo");
            Logger.Instance.Information("info 1");
            Logger.Instance.Information("info 2: {AnotherMessage:l}", "bar");

            var logger = Logger.Instance.ForContext<LoggerTests>();
            logger.Debug("debug 3");
            logger.Information("info 3");

            Assert.Equal(6, debugSink.Events.Count);
            Assert.Equal(3, infoSink.Events.Count);

            Assert.Contains("debug 2: \"foo\"", debugSink.Events.Select(e => e.RenderMessage()));
            Assert.Contains("info 2: bar", debugSink.Events.Select(e => e.RenderMessage()));
        }
    }
}
