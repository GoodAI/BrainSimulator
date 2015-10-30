using System;
using System.Collections;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Reflection;
using Serilog.Core;
using Serilog.Events;

namespace Serilog.Exceptions
{
    /// <summary>
    /// Enrich a  <see cref="LogEvent"/> with details about an <see cref="LogEvent.Exception"/> if present.
    /// </summary>
    public sealed class ExceptionEnricher : ILogEventEnricher
    {
        /// <summary>
        /// List all destricturers used by default when you call <see cref="ExceptionEnricher"/> without explictely precise them.
        /// </summary>
        public static readonly IExceptionDestructurer[] DefaultDestructurers =
        {
            new ExceptionDestructurer(),
            new AggregateExceptionDestructurer(),
            new ReflectionTypeLoadExceptionDestructurer(),
            //new SqlExceptionDestructurer()
        };

        /// <summary>
        /// <see cref="ExceptionEnricher"/> instance using reflection.
        /// </summary>
        public static readonly ExceptionEnricher ReflectionBased = new ExceptionEnricher(new ReflectionBasedDestructurer());

        private readonly IEnumerable<IExceptionDestructurer> destructurers;

        /// <summary />
        public ExceptionEnricher(params IExceptionDestructurer[] destructurers)
            : this((IEnumerable<IExceptionDestructurer>)destructurers)
        {
        }

        /// <summary />
        public ExceptionEnricher(IEnumerable<IExceptionDestructurer> destructurers)
        {
            this.destructurers = destructurers ?? DefaultDestructurers;
        }

        /// <summary />
        [CLSCompliant(false)]
        public void Enrich(LogEvent logEvent, ILogEventPropertyFactory propertyFactory)
        {
            if (logEvent.Exception != null)
            {
                logEvent.AddPropertyIfAbsent(propertyFactory.CreateProperty("ExceptionDetail", this.DestructureException(logEvent.Exception), true));
            }
        }

        private Dictionary<string, object> DestructureException(Exception exception)
        {
            var data = new Dictionary<string, object>();

            foreach (var destructurer in ExceptionTypeChain(exception.GetType()).SelectMany(type => this.destructurers.Where(destructurer => destructurer.TargetType == type)))
            {
                destructurer.Destructure(exception, data, this.DestructureException);
            }

            return data;
        }

        private static IEnumerable<Type> ExceptionTypeChain(Type finalType)
        {
            var chain = new List<Type>();

            while (finalType != null && finalType != typeof(Exception) && finalType != typeof(object))
            {
                chain.Add(finalType);
                finalType = finalType.BaseType;
            }

            if (finalType != null)
            {
                chain.Add(finalType);
            }

            return chain;
        }
    }

    public interface IExceptionDestructurer
    {
        Type TargetType { get; }

        void Destructure(Exception exception, IDictionary<string, object> data, Func<Exception, IDictionary<string, object>> innerDestructure);
    }

    #region ReflectionBased destructurer

    public class ReflectionBasedDestructurer : IExceptionDestructurer
    {
        private const int MaxRecursiveLevel = 10;

        public Type TargetType
        {
            get { return typeof(Exception); }
        }

        public void Destructure(Exception exception, IDictionary<string, object> data, Func<Exception, IDictionary<string, object>> destructureException)
        {
            foreach (var p in this.DestructureObject(exception, exception.GetType(), 0))
            {
                data.Add(p.Key, p.Value);
            }
        }

        private object DestructureValue(object value, int level)
        {
            if (value == null)
            {
                return null;
            }

            var valueType = value.GetType();

            if (valueType.IsSubclassOf(typeof(MemberInfo)))
            {
                return value;
            }

            if (Type.GetTypeCode(valueType) != TypeCode.Object || valueType.IsValueType)
            {
                return value;
            }

            if (level >= MaxRecursiveLevel)
            {
                return value;
            }

            if (typeof(IDictionary).IsAssignableFrom(valueType))
            {
                return ((IDictionary)value)
                    .Cast<DictionaryEntry>()
                    .Where(e => e.Key is string)
                    .ToDictionary(e => (string)e.Key, e => this.DestructureValue(e.Value, level + 1));
            }

            if (typeof(IEnumerable).IsAssignableFrom(valueType))
            {
                return ((IEnumerable)value)
                    .Cast<object>()
                    .Select(o => this.DestructureValue(o, level + 1))
                    .ToList();
            }

            return this.DestructureObject(value, valueType, level);
        }

        private IDictionary<string, object> DestructureObject(object value, Type valueType, int level)
        {
            var values = valueType
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(p => p.CanRead)
                .ToDictionary(p => p.Name, p => this.DestructureValue(p.GetValue(value), level + 1));

            values.Add("$Type", valueType);

            return values;
        }
    }

    #endregion

    #region Explicit destructurers

    public class ExceptionDestructurer : IExceptionDestructurer
    {
        public Type TargetType
        {
            get { return typeof(Exception); }
        }

        public void Destructure(Exception exception, IDictionary<string, object> data, Func<Exception, IDictionary<string, object>> innerDestructure)
        {
            data.Add("$Type", exception.GetType().FullName);

            if (exception.Data.Count != 0)
            {
                data.Add("Data", exception.Data.Cast<DictionaryEntry>().Where(k => k.Key is string).ToDictionary(e => (string)e.Key, e => e.Value));
            }

            if (!string.IsNullOrEmpty(exception.HelpLink))
            {
                data.Add("HelpLink", exception.HelpLink);
            }

            if (exception.HResult != 0)
            {
                data.Add("HResult", exception.HResult);
            }

            data.Add("Message", exception.Message);
            data.Add("Source", exception.Source);
            data.Add("StackTrace", exception.StackTrace);
            data.Add("TargetSite", exception.TargetSite.ToString());

            if (exception.InnerException != null)
            {
                data.Add("InnerException", innerDestructure(exception.InnerException));
            }
        }
    }

    public class AggregateExceptionDestructurer : IExceptionDestructurer
    {
        public Type TargetType
        {
            get { return typeof(AggregateException); }
        }

        public void Destructure(Exception exception, IDictionary<string, object> data, Func<Exception, IDictionary<string, object>> destructureException)
        {
            var aggEx = (AggregateException)exception;

            data.Add("InnerExceptions", aggEx.InnerExceptions.Select(destructureException).ToList());
        }
    }

    public class ReflectionTypeLoadExceptionDestructurer : IExceptionDestructurer
    {
        public Type TargetType
        {
            get { return typeof(ReflectionTypeLoadException); }
        }

        public void Destructure(Exception exception, IDictionary<string, object> data, Func<Exception, IDictionary<string, object>> destructureException)
        {
            var tle = (ReflectionTypeLoadException)exception;

            if (tle.LoaderExceptions != null)
            {
                data.Add("LoaderExceptions", tle.LoaderExceptions.Select(destructureException).ToList());
            }
        }
    }

    //public class SqlExceptionDestructurer : IExceptionDestructurer
    //{
    //    public Type TargetType
    //    {
    //        get { return typeof(SqlException); }
    //    }

    //    public void Destructure(Exception exception, IDictionary<string, object> data, Func<Exception, IDictionary<string, object>> destructureException)
    //    {
    //        var sqlEx = (System.Data.SqlClient.SqlException)exception;

    //        data.Add("ClientConnectionId", sqlEx.ClientConnectionId);
    //        data.Add("Class", sqlEx.Class);
    //        data.Add("LineNumber", sqlEx.LineNumber);
    //        data.Add("Number", sqlEx.Number);
    //        data.Add("Server", sqlEx.Server);
    //        data.Add("State", sqlEx.State);
    //        data.Add("Errors", sqlEx.Errors
    //            .Cast<System.Data.SqlClient.SqlError>()
    //            //.Select(sqlErr => new { sqlErr.Class, sqlErr.LineNumber, sqlErr.Message, sqlErr.Number, sqlErr.Procedure, sqlErr.Server, sqlErr.Source, sqlErr.State })
    //            .ToList());
    //    }
    //}

    #endregion
}
