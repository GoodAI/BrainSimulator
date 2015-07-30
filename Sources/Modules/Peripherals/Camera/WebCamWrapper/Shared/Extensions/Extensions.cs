using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media.Imaging;
using WindowsPoint = System.Windows.Point;
using DrawingPoint = System.Drawing.Point;

namespace Touchless.Shared.Extensions
{
    public static class Extensions
    {
        public static IEnumerable<T> ForEach<T>(this IEnumerable<T> items, Action<T> action)
        {
            foreach (T item in items)
            {
                action(item);
            }

            return items;
        }

        public static void IfNotNull<T>(this T item, Action<T> action)
        {
            if (item != null)
            {
                action(item);
            }
        }

        public static WindowsPoint ToWindowsPoint(this DrawingPoint p)
        {
            return new WindowsPoint
                       {
                           X = p.X,
                           Y = p.Y
                       };
        }

        public static DrawingPoint ToDrawingPoint(this WindowsPoint p)
        {
            return new DrawingPoint
                       {
                           X = (int) p.X,
                           Y = (int) p.Y
                       };
        }

        [DllImport("gdi32")]
        private static extern int DeleteObject(IntPtr o);

        public static BitmapSource ToBitmapSource(this Bitmap source)
        {
            BitmapSource bs = null;

            IntPtr ip = source.GetHbitmap();
            try
            {
                bs = Imaging.CreateBitmapSourceFromHBitmap(ip, IntPtr.Zero, Int32Rect.Empty,
                                                           BitmapSizeOptions.FromEmptyOptions());
            }
            finally
            {
                DeleteObject(ip);
            }

            return bs;
        }
    }
}