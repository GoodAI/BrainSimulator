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
            // TODO: more sensible values
            return new SizeF(GraphConstants.MinimumItemWidth, GraphConstants.MinimumItemHeight);
        }

        internal override void Render(Graphics graphics, SizeF minimumSize, PointF position)
        {
            SizeF size = Measure(graphics);
            size.Width = Math.Max(minimumSize.Width, size.Width);
            size.Height = Math.Max(minimumSize.Height, size.Height);

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

        private void DrawTextItem(Graphics graphics, PointF position, SizeF size)
        {
            var font = new Font(SystemFonts.MenuFont.FontFamily, SystemFonts.MenuFont.Size - 2.0f);

            graphics.DrawString(TextItem, font, Brushes.Black,
                new RectangleF(position, size),
                GraphConstants.RightTextStringFormat);
        }
    }
}
