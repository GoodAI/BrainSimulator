#region License
// Copyright (c) 2009 Sander van Rossen
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Drawing;

namespace Graph.Items
{
	public class NodeLabelItem : NodeItem
	{
		public NodeLabelItem(string text, bool inputEnabled, bool outputEnabled) :
			base(inputEnabled, outputEnabled)
		{
			this.Text = text;
		}

		public NodeLabelItem(string text) :
			this(text, false, false) { }

		#region Text
		protected string internalText = string.Empty;
		public virtual string Text
		{
			get { return internalText; }
			set
			{
				if (internalText == value)
					return;

				internalText = value;
				textSize = Size.Empty;
			}
		}
		#endregion

		private SizeF textSize;

		private StringFormat GetTextFormat()
		{
			return (this.Input.Enabled == this.Output.Enabled)
				? GraphConstants.CenterMeasureTextStringFormat
				: (this.Input.Enabled
					? GraphConstants.LeftMeasureTextStringFormat
					: GraphConstants.RightMeasureTextStringFormat);
		}

		internal override SizeF Measure(Graphics graphics)
		{
			// HACK: MeasureString does not work for unicode characters like "⊕", increase min. height
			float minTextHeight = Math.Max(16.0f, GraphConstants.MinimumItemHeight); 

			if (!string.IsNullOrWhiteSpace(this.Text))
			{
				if (this.textSize.IsEmpty)
				{
					var size = new SizeF(GraphConstants.MinimumItemWidth, minTextHeight);

					this.textSize = graphics.MeasureString(this.Text, SystemFonts.MenuFont, size, GetTextFormat());

					this.textSize.Width  = Math.Max(size.Width, this.textSize.Width);
					this.textSize.Height = Math.Max(size.Height, this.textSize.Height);
				}

				return this.textSize;
			}
			else
			{
				return new SizeF(GraphConstants.MinimumItemWidth, GraphConstants.MinimumItemHeight);
			}
		}

		internal override void Render(Graphics graphics, SizeF minimumSize, PointF location)
		{
			var size = Measure(graphics);
			size.Width  = Math.Max(minimumSize.Width, size.Width);
			size.Height = Math.Max(minimumSize.Height, size.Height);

			graphics.DrawString(this.Text, SystemFonts.MenuFont, Brushes.DarkSlateGray, new RectangleF(location, size), GetTextFormat());
		}
	}
}
