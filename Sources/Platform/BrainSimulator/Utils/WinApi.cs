using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

// Some code adapted from www.pinvoke.net

namespace GoodAI.BrainSimulator.Utils
{
    /// <summary>
    /// Wrapper around the Winapi POINT type.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT
    {
        public int X;
        public int Y;

        public POINT(int x, int y)
        {
            X = x;
            Y = y;
        }

        /// <summary>Implicit cast.</summary>
        public static implicit operator Point(POINT p)
        {
            return new Point(p.X, p.Y);
        }

        /// <summary> Implicit cast.</summary>
        public static implicit operator POINT(Point p)
        {
            return new POINT(p.X, p.Y);
        }
    }

    /// <summary>
    /// Wrapper around the Winapi RECT type.
    /// </summary>
    [Serializable, StructLayout(LayoutKind.Sequential)]
    public struct RECT
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;

        public RECT(int left, int top, int right, int bottom)
        {
            Left = left;
            Top = top;
            Right = right;
            Bottom = bottom;
        }

        public int Height => Bottom - Top;
        public int Width => Right - Left;
        public Size Size => new Size(Width, Height);
        public Point Location => new Point(Left, Top);

        /// <summary>Convert RECT to a Rectangle.</summary>
        public Rectangle ToRectangle()
        {
            return Rectangle.FromLTRB(Left, Top, Right, Bottom);
        }

        /// <summary>Convert Rectangle to a RECT</summary>
        public static RECT FromRectangle(Rectangle rectangle)
        {
            return new RECT(rectangle.Left, rectangle.Top, rectangle.Right, rectangle.Bottom);
        }

        #region Implicit casts

        /// <summary>Implicit Cast.</summary>
        public static implicit operator Rectangle(RECT rect)
        {
            return Rectangle.FromLTRB(rect.Left, rect.Top, rect.Right, rect.Bottom);
        }

        /// <summary>Implicit Cast.</summary>
        public static implicit operator RECT(Rectangle rect)
        {
            return new RECT(rect.Left, rect.Top, rect.Right, rect.Bottom);
        }

        #endregion
    }

    public struct WindowPlacement
    {
        public FormWindowState WindowState;

        /// <summary>A rectangle describing both location and size.</summary>
        public Rectangle Position;

        public WindowPlacement(FormWindowState windowState, Rectangle position)
        {
            WindowState = windowState;
            Position = position;
        }
    }

    public static class WinApi
    {
        public static WindowPlacement GetWindowPlacement(Control form)
        {
            return GetWindowPlacement(form.Handle);
        }

        public static WindowPlacement GetWindowPlacement(IntPtr windowHandle)
        {
            var wp = new WINDOWPLACEMENT();
            wp.length = Marshal.SizeOf(wp);

            var retVal = GetWindowPlacement(windowHandle, ref wp);
            if (!retVal)
                throw new Win32Exception("Call to GetWindowPlacement failed");

            return new WindowPlacement(ShowCmdToWindowState(wp.showCmd), wp.rcNormalPosition);
        }

        public static void SetWindowPlacement(Control form, FormWindowState windowState, Rectangle position)
        {
            SetWindowPlacement(form.Handle, windowState, position);
        }

        public static void SetWindowPlacement(IntPtr windowHandle, FormWindowState windowState, Rectangle position)
        {
            var wp = new WINDOWPLACEMENT();
            wp.length = Marshal.SizeOf(wp);

            var retVal = GetWindowPlacement(windowHandle, ref wp);
            if (!retVal)
                throw new Win32Exception("Call to GetWindowPlacement failed");

            wp.showCmd = WindowStateToShowCmd(windowState);
            wp.rcNormalPosition = position;

            retVal = SetWindowPlacement(windowHandle, ref wp);
            if (!retVal)
                throw new Win32Exception("Call to SetWindowPlacement failed");
        }

        private static FormWindowState ShowCmdToWindowState(int showCmd)
        {
            switch (showCmd % 4)
            {
                case 2: return FormWindowState.Minimized;
                case 3: return FormWindowState.Maximized;
                default: return FormWindowState.Normal;
            }
        }

        private static int WindowStateToShowCmd(FormWindowState windowState)
        {
            switch (windowState)
            {
                case FormWindowState.Normal: return 1;
                case FormWindowState.Minimized: return 2;
                case FormWindowState.Maximized: return 3;
                default: throw new ArgumentOutOfRangeException(nameof(windowState), "Invalid value");
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct WINDOWPLACEMENT
        {
            public int length;
            public int flags;
            public int showCmd;
            public POINT ptMinPosition;
            public POINT ptMaxPosition;
            public RECT rcNormalPosition;
        }

        [DllImport("user32.dll")]
        private static extern bool GetWindowPlacement(IntPtr hWnd, ref WINDOWPLACEMENT lpwndpl);

        [DllImport("user32.dll")]
        private static extern bool SetWindowPlacement(IntPtr hWnd, [In] ref WINDOWPLACEMENT lpwndpl);
    }
}
