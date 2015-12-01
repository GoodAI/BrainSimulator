using Graph;
using Graph.Items;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.BrainSimulator.Nodes
{
    internal class MyStatusBarItem : NodeItem
    {
        public class IconSubitem
        {
            public IconSubitem(Image image, bool enabled)
            {
                Image = image;
                Enabled = enabled;
            }

            public Image Image { get; private set; }
            public bool Enabled { get; set; }
        }

        public MyStatusBarItem() : base(enableInput: false, enableOutput: false)
        {
            IsPassive = true;

            IconSubitems = new List<IconSubitem>();
            IconSize = new SizeF(12.0f, 12.0f);
        }

        public string TextItem { get; set; }

        public IList<IconSubitem> IconSubitems { get; private set; }

        public SizeF IconSize { get; set; }

        private const float PaddingF = 2.0f;

        public IconSubitem AddIcon(Image image, bool enabled = false)
        {
            var icon = new IconSubitem(image, enabled);

            IconSubitems.Add(icon);

            return icon;
        }

        internal override SizeF Measure(Graphics context)
        {
            var size = new SizeF(GraphConstants.MinimumItemWidth, GraphConstants.MinimumItemHeight);

            var textSize = DrawTextItem(context, PointF.Empty, size, justMeasure: true);

            var iconSubitemsSize = (IconSubitems.Count == 0)
                ? SizeF.Empty
                : new SizeF(IconSubitems.Count() * (IconSize.Width + PaddingF), IconSize.Height);

            return new SizeF(
                Math.Max(GraphConstants.MinimumItemWidth, textSize.Width + iconSubitemsSize.Width),
                Math.Max(Math.Max(GraphConstants.MinimumItemHeight, textSize.Height), iconSubitemsSize.Height));
        }

        internal override void Render(Graphics graphics, SizeF minimumSize, PointF position)
        {
            SizeF size = Measure(graphics);
            size.Width = Math.Max(minimumSize.Width, size.Width);
            size.Height = Math.Max(minimumSize.Height, size.Height);

            position.Y += 4;
            DrawIconSubitems(graphics, position);

            DrawTextItem(graphics, position, size);
        }

        private void DrawIconSubitems(Graphics graphics, PointF position)
        {
            foreach (IconSubitem icon in IconSubitems)
            {
                if (!icon.Enabled)
                    continue;

                graphics.DrawImage(icon.Image, new RectangleF(position, IconSize));

                position.X += IconSize.Width + PaddingF;
            }
        }

        private SizeF DrawTextItem(Graphics graphics, PointF position, SizeF size, bool justMeasure = false)
        {
            var font = new Font(SystemFonts.MenuFont.FontFamily, SystemFonts.MenuFont.Size - 2.0f, FontStyle.Bold);
            var text = TextItem;
            var format = GraphConstants.RightMeasureTextStringFormat;

            return DrawOrMeasureString(graphics, text, font, position, size, format, justMeasure);
        }

        // TODO(Premek): Move to up to the library (or to a different lib).
        private static SizeF DrawOrMeasureString(Graphics graphics, string text, Font font, PointF position, SizeF size,
            StringFormat format, bool measure)
        {
            if (measure)
            {
                return graphics.MeasureString(text, font, size, format);
            }
            else
            {
                graphics.DrawString(text, font, Brushes.DimGray, new RectangleF(position, size), format);
                return SizeF.Empty;
            }
        }
    }
}
