using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

namespace GoodAI.BrainSimulator
{
    /// <summary>
    /// Provides handers for unhandled exceptions for WinForms Application.
    /// 
    /// Currently only shows a message box with exception details. No logging (yet).
    /// 
    /// See: http://stackoverflow.com/questions/5762526/how-can-i-make-something-that-catches-all-unhandled-exceptions-in-a-winforms-a
    /// </summary>
    internal static class UnhandledExceptionCatcher
    {
        public static void RegisterHandlers()
        {
            if (!Debugger.IsAttached)
            {
                Application.ThreadException += new ThreadExceptionEventHandler(ProcessThreadException);

                Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);

                AppDomain.CurrentDomain.UnhandledException += new UnhandledExceptionEventHandler(ProcessUnhandledException);
            }
        }

        static void ProcessThreadException(object sender, ThreadExceptionEventArgs e)
        {
            ShowExceptionAndExit("Unhandled Thread Exception", e.Exception);
        }

        static void ProcessUnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            ShowExceptionAndExit("Unhandled UI Exception", e.ExceptionObject as Exception);
        }

        static void ShowExceptionAndExit(string title, Exception ex)
        {
            MessageBox.Show("Unhandled exception encountered, sorry :-("
                + "\n\n" + PrintException(ex),
                title, MessageBoxButtons.OK, MessageBoxIcon.Error);

            Environment.Exit(1);
        }

        static string PrintException(Exception ex, bool isInner = false)
        {
            return (ex == null)
                ? "(null)"
                : "Message:\n" + ex.Message
                    // Print detiled stack trace for the first and last exception in the chain.
                    + PrintStackTrace(ex.StackTrace, brief: (isInner && ex.InnerException != null))
                    + (ex.InnerException != null
                        ? "\n\nInner Exception " + PrintException(ex.InnerException, isInner: true)
                        : "");
        }

        static string PrintStackTrace(string trace, bool brief = false)
        {
            if (string.IsNullOrEmpty(trace))
                return "";

            return brief
                ? "\n" + trace.Split('\n', '\r').FirstOrDefault()
                : "\n\nStack trace:\n" + trace;
        }
    }
}
