﻿#region License
// Copyright (c) 2009 Sander van Rossen, 2013 Oliver Salzburg
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
using System.Drawing;
using System.Drawing.Drawing2D;

namespace Graph
{
    public static class GraphRenderer
    {
        private static GraphStyle m_defaultStyle = null;
        public static GraphStyle DefaultStyle
        {
            get
            {
                if (m_defaultStyle == null)
                {
                    m_defaultStyle = new GraphStyle();
                }
                return m_defaultStyle;
            }
        }

        static IEnumerable<NodeItem> EnumerateNodeItems(Node node)
        {
            if (node == null)
                yield break;

            yield return node.titleItem;
            if (node.Collapsed)
            {               
                yield break;
            }            
            
            foreach (var item in node.Items)
                yield return item;
        }

        public static SizeF Measure(Graphics context, Node node)
        {
            if (node == null)
                return SizeF.Empty;

            SizeF size = Size.Empty;
            size.Height = //(int)NodeConstants.TopHeight + 
                (int)GraphConstants.BottomHeight;
            foreach (var item in EnumerateNodeItems(node))
            {
                var itemSize = item.Measure(context);
                size.Width = Math.Max(size.Width, itemSize.Width);
                size.Height += GraphConstants.ItemSpacing + itemSize.Height;
            }
            
            if (node.Collapsed)
                size.Height -= GraphConstants.ItemSpacing;

            size.Width += GraphConstants.NodeExtraWidth;
            return size;
        }

        static SizeF PreRenderItem(Graphics graphics, NodeItem item, PointF position)
        {
            var itemSize = (SizeF)item.Measure(graphics);
            item.bounds = new RectangleF(position, itemSize);
            return itemSize;
        }

        static void RenderItem(Graphics graphics, SizeF minimumSize, NodeItem item, PointF position)
        {
            item.Render(graphics, minimumSize, position);
        }

        private static Pen BorderPen = new Pen(Color.FromArgb(64, 64, 64));

        static void RenderConnector(Graphics graphics, RectangleF bounds, RenderState state)
        {
            using (var brush = new SolidBrush(GetArrowLineColor(state)))
            {
                graphics.FillEllipse(brush, bounds);
            }
            
            if (state == RenderState.None)
            {
                graphics.DrawEllipse(Pens.Black, bounds);
            } else
            // When we're compatible, but not dragging from this node we render a highlight
            if ((state & (RenderState.Compatible | RenderState.Dragging)) == RenderState.Compatible) 
            {
                // First draw the normal black border
                graphics.DrawEllipse(Pens.Black, bounds);

                // Draw an additional highlight around the connector
                RectangleF highlightBounds = new RectangleF(bounds.X,bounds.Y,bounds.Width,bounds.Height);
                highlightBounds.Width += 10;
                highlightBounds.Height += 10;
                highlightBounds.X -= 5;
                highlightBounds.Y -= 5;
                graphics.DrawEllipse(Pens.OrangeRed, highlightBounds);
            } else
            {
                graphics.DrawArc(Pens.Black, bounds, 90, 180);
                using (var pen = new Pen(GetArrowLineColor(state)))
                {
                    graphics.DrawArc(pen, bounds, 270, 180);
                }
            }			
        }

        static void RenderArrow(Graphics graphics, RectangleF bounds, RenderState connectionState)
        {
            var x = (bounds.Left + bounds.Right) / 2.0f;
            var y = (bounds.Top + bounds.Bottom) / 2.0f;
            using (var brush = new SolidBrush(GetArrowLineColor(connectionState | RenderState.Connected)))
            {
                graphics.FillPolygon(brush, DefaultStyle.GetArrowPoints(x,y), FillMode.Winding);
            }
        }

        public static void PerformLayout(Graphics graphics, IEnumerable<Node> nodes)
        {
            foreach (var node in nodes.Reverse<Node>())
            {
                GraphRenderer.PerformLayout(graphics, node);
            }
        }

        public static void Render(Graphics graphics, IEnumerable<Node> nodes, bool showLabels)
        {
            var skipConnections = new HashSet<NodeConnection>();
            foreach (var node in nodes.Reverse<Node>())
            {
                GraphRenderer.RenderConnections(graphics, node, skipConnections, showLabels);
            }
            foreach (var node in nodes.Reverse<Node>())
            {
                GraphRenderer.Render(graphics, node);
            }
        }

        public static void PerformLayout(Graphics graphics, Node node)
        {
            if (node == null)
                return;
            var size		= Measure(graphics, node);
            var position	= node.Location;
            node.bounds		= new RectangleF(position, size);
            
            var path				= new GraphicsPath(FillMode.Winding);
            int connectorSize		= (int)GraphConstants.ConnectorSize;
            int halfConnectorSize	= (int)Math.Ceiling(connectorSize / 2.0f);
            var connectorOffset		= (int)Math.Floor((GraphConstants.MinimumItemHeight - GraphConstants.ConnectorSize) / 2.0f);
            var left				= position.X + halfConnectorSize;
            var top					= position.Y;
            var right				= position.X + size.Width - halfConnectorSize;
            var bottom				= position.Y + size.Height;
            
            node.inputConnectors.Clear();
            node.outputConnectors.Clear();
            //node.connections.Clear();

            var itemPosition = position;
            itemPosition.X += connectorSize + (int)GraphConstants.HorizontalSpacing;
            if (node.Collapsed)
            {
                foreach (var item in node.Items)
                {
                    var inputConnector	= item.Input;
                    if (inputConnector != null && inputConnector.Enabled)
                    {
                        inputConnector.bounds = Rectangle.Empty;
                        node.inputConnectors.Add(inputConnector);
                    }
                    var outputConnector = item.Output;
                    if (outputConnector != null && outputConnector.Enabled)
                    {
                        outputConnector.bounds = Rectangle.Empty;
                        node.outputConnectors.Add(outputConnector);
                    }
                }
                var itemSize		= PreRenderItem(graphics, node.titleItem, itemPosition);
                var realHeight		= itemSize.Height - GraphConstants.TopHeight;
                var connectorY		= itemPosition.Y  + (int)Math.Ceiling(realHeight / 2.0f);
                
                node.inputBounds	= new RectangleF(left  - (GraphConstants.ConnectorSize / 2), 
                                                     connectorY, 
                                                     GraphConstants.ConnectorSize, 
                                                     GraphConstants.ConnectorSize);
                node.outputBounds	= new RectangleF(right - (GraphConstants.ConnectorSize / 2), 
                                                     connectorY, 
                                                     GraphConstants.ConnectorSize, 
                                                     GraphConstants.ConnectorSize);
            } else
            {
                node.inputBounds	= Rectangle.Empty;
                node.outputBounds	= Rectangle.Empty;
                
                foreach (var item in EnumerateNodeItems(node))
                {
                    var itemSize		= PreRenderItem(graphics, item, itemPosition);
                    var realHeight		= itemSize.Height;
                    var inputConnector	= item.Input;
                    if (inputConnector != null && inputConnector.Enabled)
                    {
                        if (itemSize.IsEmpty)
                        {
                            inputConnector.bounds = Rectangle.Empty;
                        } else
                        {
                            inputConnector.bounds = new RectangleF(	left - (GraphConstants.ConnectorSize / 2),
                                                                    itemPosition.Y + realHeight * 0.5f - GraphConstants.ConnectorSize / 2, //connectorOffset, 
                                                                    GraphConstants.ConnectorSize, 
                                                                    GraphConstants.ConnectorSize);
                        }
                        node.inputConnectors.Add(inputConnector);
                    }
                    var outputConnector = item.Output;
                    if (outputConnector != null && outputConnector.Enabled)
                    {
                        if (itemSize.IsEmpty)
                        {
                            outputConnector.bounds = Rectangle.Empty;
                        } else
                        {
                            outputConnector.bounds = new RectangleF(right - (GraphConstants.ConnectorSize / 2), 
                                                                    itemPosition.Y + realHeight * 0.5f - GraphConstants.ConnectorSize / 2, //realHeight - (connectorOffset + GraphConstants.ConnectorSize), 
                                                                    GraphConstants.ConnectorSize, 
                                                                    GraphConstants.ConnectorSize);
                        }
                        node.outputConnectors.Add(outputConnector);
                    }
                    itemPosition.Y += itemSize.Height + GraphConstants.ItemSpacing;
                }
            }
            node.itemsBounds = new RectangleF(left, top, right - left, bottom - top);
        }

        static Brush ResolveNodeBrush(Node node, Brush baseBrush)
        {
            if ((node.state & (RenderState.Dragging | RenderState.Focus)) != 0)
            {
                return Brushes.DarkOrange;
            }
            else if ((node.state & RenderState.Hover) != 0)
            {
                return Brushes.LightSteelBlue;
            }
            else
            {
                return baseBrush;
            }
        }

        static void Render(Graphics graphics, Node node)
        {
            if (!node.shown)
                return;

            var size		= node.bounds.Size;
            var position	= node.bounds.Location;
            
            int cornerSize			= (int)GraphConstants.CornerSize * 2;
            int connectorSize		= (int)GraphConstants.ConnectorSize;
            int halfConnectorSize	= (int)Math.Ceiling(connectorSize / 2.0f);
            var connectorOffset		= (int)Math.Floor((GraphConstants.MinimumItemHeight - GraphConstants.ConnectorSize) / 2.0f);
            var left				= position.X + halfConnectorSize;
            var top					= position.Y;
            var right				= position.X + size.Width - halfConnectorSize;
            var bottom				= position.Y + size.Height;
            using (var path = new GraphicsPath(FillMode.Winding))
            {
                path.AddArc(left, top, cornerSize, cornerSize, 180, 90);
                path.AddArc(right - cornerSize, top, cornerSize, cornerSize, 270, 90);                                

                path.AddArc(right - cornerSize, bottom - cornerSize, cornerSize, cornerSize, 0, 90);
                path.AddArc(left, bottom - cornerSize, cornerSize, cornerSize, 90, 90);
                path.CloseFigure();

                graphics.FillPath(ResolveNodeBrush(node, node.background), path);

                if (node.HasStatusBar)
                {
                    using (var pathBottomStatus = new GraphicsPath(FillMode.Winding))
                    {
                        pathBottomStatus.AddLine(left, bottom - GraphConstants.StatusBarHeight, right, bottom - GraphConstants.StatusBarHeight);

                        pathBottomStatus.AddArc(right - cornerSize, bottom - cornerSize, cornerSize, cornerSize, 0, 90);
                        pathBottomStatus.AddArc(left, bottom - cornerSize, cornerSize, cornerSize, 90, 90);
                        pathBottomStatus.CloseFigure();

                        graphics.FillPath(ResolveNodeBrush(node, new SolidBrush(Color.FromArgb(190, 194, 200))), 
                            pathBottomStatus);
                    }
                }

                graphics.DrawPath(BorderPen, path);
            }
            /*
            if (!node.Collapsed)
                graphics.DrawLine(Pens.Black, 
                    left  + GraphConstants.ConnectorSize, node.titleItem.bounds.Bottom - GraphConstants.ItemSpacing, 
                    right - GraphConstants.ConnectorSize, node.titleItem.bounds.Bottom - GraphConstants.ItemSpacing);
            */
            var itemPosition = position;
            itemPosition.X += connectorSize + (int)GraphConstants.HorizontalSpacing;
            if (node.Collapsed)
            {
                bool inputConnected = false;
                var inputState	= RenderState.None;
                var outputState = node.outputState;
                foreach (var connection in node.connections)
                {
                    if (connection.To.Node == node)
                    {
                        inputState |= connection.state;
                        inputConnected = true;
                    }
                    if (connection.From.Node == node)
                        outputState |= connection.state | RenderState.Connected;
                }

                RenderItem(graphics, new SizeF(node.bounds.Width - GraphConstants.NodeExtraWidth, 0), node.titleItem, itemPosition);
                if (node.inputConnectors.Count > 0)
                    RenderConnector(graphics, node.inputBounds, node.inputState);
                if (node.outputConnectors.Count > 0)
                    RenderConnector(graphics, node.outputBounds, outputState);
                if (inputConnected)
                    RenderArrow(graphics, node.inputBounds, inputState);
            } else
            {
                node.inputBounds	= Rectangle.Empty;
                node.outputBounds	= Rectangle.Empty;
                
                var minimumItemSize = new SizeF(node.bounds.Width - GraphConstants.NodeExtraWidth, 0);
                foreach (var item in EnumerateNodeItems(node))
                {
                    RenderItem(graphics, minimumItemSize, item, itemPosition);
                    var inputConnector	= item.Input;
                    if (inputConnector != null && inputConnector.Enabled)
                    {
                        if (!inputConnector.bounds.IsEmpty)
                        {
                            var state		= RenderState.None;
                            var connected	= false;
                            foreach (var connection in node.connections)
                            {
                                if (connection.To == inputConnector)
                                {
                                    state |= connection.state;
                                    connected = true;
                                }
                            }

                            RenderConnector(graphics, 
                                            inputConnector.bounds,
                                            inputConnector.state);

                            if (connected)
                                RenderArrow(graphics, inputConnector.bounds, state);
                        }
                    }
                    var outputConnector = item.Output;
                    if (outputConnector != null && outputConnector.Enabled)
                    {
                        if (!outputConnector.bounds.IsEmpty)
                        {
                            var state = outputConnector.state;
                            foreach (var connection in node.connections)
                            {
                                if (connection.From == outputConnector)
                                    state |= connection.state | RenderState.Connected;
                            }
                            RenderConnector(graphics, outputConnector.bounds, state);
                        }
                    }
                    itemPosition.Y += item.bounds.Height + GraphConstants.ItemSpacing;
                }
            }
        }

        public static void RenderConnections(Graphics graphics, Node node, HashSet<NodeConnection> skipConnections, bool showLabels)
        {
            if (!node.shown)
                return;            

            foreach (var connection in node.connections.Reverse<NodeConnection>())
            {
                if (connection == null ||
                    connection.From == null ||
                    connection.To == null ||
                    !connection.From.Node.shown ||
                    !connection.To.Node.shown)
                    continue;

                if (skipConnections.Add(connection))
                {
                    var to		= connection.To;
                    var from	= connection.From;
                    RectangleF toBounds;
                    RectangleF fromBounds;
                    if (to.Node.Collapsed)		toBounds = to.Node.inputBounds;
                    else						toBounds = to.bounds;
                    if (from.Node.Collapsed)	fromBounds = from.Node.outputBounds;
                    else						fromBounds = from.bounds;

                    var x1 = (fromBounds.Left + fromBounds.Right) / 2.0f;
                    var y1 = (fromBounds.Top + fromBounds.Bottom) / 2.0f;
                    var x2 = (toBounds.Left + toBounds.Right) / 2.0f;
                    var y2 = (toBounds.Top + toBounds.Bottom) / 2.0f;

                    float centerX;
                    float centerY;
                    int xOffset = GraphConstants.HiddenConnectionLabelOffset;

                    bool isFromNodeHover = (connection.From.Node.state & RenderState.Hover) != 0;
                    bool isConnectionHidden = (connection.state & RenderState.Hidden) != 0;
                    bool isConnectionHover = (connection.state & RenderState.Hover) != 0;

                    using (var path = GetArrowLinePath(x1, y1, x2, y2, out centerX, out centerY, false))
                    {
                        Color arrowLineColor = GetArrowLineColor(connection.state | RenderState.Connected);

                        if ((connection.state & RenderState.Marked) != 0)
                        {
                            Color glowColorBase = Color.Ivory;
                            Color glowColor = Color.FromArgb(
                                (int)(glowColorBase.A * 0.5),
                                glowColorBase.R,
                                glowColorBase.G,
                                glowColorBase.B);
                            // Draw a glow.
                            var pen = new Pen(new SolidBrush(glowColor), 4.0f);
                            graphics.DrawPath(pen, path);
                        }

                        Brush brush = new SolidBrush(arrowLineColor);

                        if (isConnectionHidden && !isConnectionHover)
                        {
                            if (isFromNodeHover)
                            {
                                graphics.FillPath(new SolidBrush(GetArrowLineColor(RenderState.Hover)), path);                                
                            }
                            else
                            {
                                graphics.FillRectangle(brush, x1, y1 - 0.75f, xOffset, 1.5f);
                                graphics.FillRectangle(brush, x2 - xOffset, y2 - 2, xOffset, 4);
                            }
                        }
                        else
                        {
                            graphics.FillPath(brush, path);
                        }

                        connection.bounds = RectangleF.Union(path.GetBounds(), connection.textBounds);                        
                    }

                    if (showLabels && !string.IsNullOrWhiteSpace(connection.Name))
                    {
                        if (isConnectionHidden)
                        {
                            RenderState rState = isFromNodeHover ? connection.state | RenderState.Hover : connection.state;                            

                            var center = new PointF(x1, y1);
                            RenderLabel(graphics, connection, center, rState, true);

                            center = new PointF(x2, y2);
                            RenderLabel(graphics, connection, center, rState);
                        }
                        else
                        {
                            var center = new PointF(centerX, centerY);
                            RenderLabel(graphics, connection, center, connection.state);
                        }
                    }
                }
            }
        }

        static void RenderLabel(Graphics graphics, NodeConnection connection, PointF center, RenderState state, bool isFromLabel = false)
        {
            using (var path = new GraphicsPath(FillMode.Winding))
            {			
                int cornerSize			= (int)GraphConstants.CornerSize * 2;
                int connectorSize		= (int)GraphConstants.ConnectorSize;
                int halfConnectorSize	= (int)Math.Ceiling(connectorSize / 2.0f);
                int xOffset = GraphConstants.HiddenConnectionLabelOffset;

                bool isConnectionHidden = (connection.state & RenderState.Hidden) != 0;

                SizeF size;
                PointF position;
                PointF textPosition = center;

                var text = isConnectionHidden && isFromLabel ? "⋯" : connection.Name;

                if (connection.textBounds.IsEmpty ||
                    connection.textBounds.Location != center)
                {
                    size = graphics.MeasureString(text, SystemFonts.StatusFont, center, GraphConstants.CenterTextStringFormat);

                    if (isConnectionHidden) 
                    {
                        if (isFromLabel)
                        {
                            textPosition = new PointF(center.X + xOffset + size.Width / 2.0f, center.Y);
                            position = new PointF(center.X + xOffset - halfConnectorSize, center.Y - (size.Height / 2.0f));
                        }
                        else
                        {
                            textPosition = new PointF(center.X - xOffset - size.Width / 2.0f, center.Y);
                            position = new PointF(center.X - xOffset - size.Width - halfConnectorSize, center.Y - (size.Height / 2.0f));
                        }
                    }
                    else
                    {                        
                        position = new PointF(center.X - (size.Width / 2.0f) - halfConnectorSize, center.Y - (size.Height / 2.0f));
                    }

                    size.Width	+= connectorSize;
                    connection.textBounds = new RectangleF(position, size);
                } 
                else
                {
                    size		= connection.textBounds.Size;
                    position	= connection.textBounds.Location;
                }

                var halfWidth  = size.Width / 2.0f;
                var halfHeight = size.Height / 2.0f;
                var connectorOffset		= (int)Math.Floor((GraphConstants.MinimumItemHeight - GraphConstants.ConnectorSize) / 2.0f);
                var left				= position.X;
                var top					= position.Y;
                var right				= position.X + size.Width;
                var bottom				= position.Y + size.Height;
                path.AddArc(left, top, cornerSize, cornerSize, 180, 90);
                path.AddArc(right - cornerSize, top, cornerSize, cornerSize, 270, 90);

                path.AddArc(right - cornerSize, bottom - cornerSize, cornerSize, cornerSize, 0, 90);
                path.AddArc(left, bottom - cornerSize, cornerSize, cornerSize, 90, 90);
                path.CloseFigure();

                using (var brush = new SolidBrush(GetArrowLineColor(state)))
                {
                    graphics.FillPath(brush, path);
                }
                graphics.DrawString(text, SystemFonts.StatusFont, Brushes.Black, textPosition, GraphConstants.CenterTextStringFormat);

                //draw outline for all conn. labels when not dragged focused or hovered
                if ((state & ~RenderState.Hidden & ~RenderState.Backward) == RenderState.None) 
                {
                    graphics.DrawPath(new Pen(GetArrowLineColor(state | RenderState.Connected)), path);
                }                

                //graphics.DrawRectangle(Pens.Red, connection.textBounds.Left, connection.textBounds.Top, connection.textBounds.Width, connection.textBounds.Height);
            }
        }

        public static Region GetConnectionRegion(NodeConnection connection)
        {
            var to		= connection.To;
            var from	= connection.From;
            RectangleF toBounds;
            RectangleF fromBounds;
            if (to.Node.Collapsed)		toBounds = to.Node.inputBounds;
            else						toBounds = to.bounds;
            if (from.Node.Collapsed)	fromBounds = from.Node.outputBounds;
            else						fromBounds = from.bounds;

            var x1 = (fromBounds.Left + fromBounds.Right) / 2.0f;
            var y1 = (fromBounds.Top + fromBounds.Bottom) / 2.0f;
            var x2 = (toBounds.Left + toBounds.Right) / 2.0f;
            var y2 = (toBounds.Top + toBounds.Bottom) / 2.0f;

            Region region;
            float centerX;
            float centerY;

            if ((connection.state & RenderState.Hidden) != 0)
            {
                region = new Region(connection.textBounds);
            }
            else
            {
                using (var linePath = GetArrowLinePath(x1, y1, x2, y2, out centerX, out centerY, true, 5.0f))
                {
                    region = new Region(linePath);
                }
            }
            return region;
        }

        static Color GetArrowLineColor(RenderState state)
        {
            if ((state & (RenderState.Hover | RenderState.Dragging)) != 0)
            {
                if ((state & RenderState.Incompatible) != 0)
                {
                    return Color.Red;
                } else
                if ((state & RenderState.Compatible) != 0)
                {
                    return Color.DarkOrange;
                } else
                if ((state & RenderState.Dragging) != 0)
                    return Color.SteelBlue;
                else
                    return Color.DarkOrange;
            } else
            if ((state & RenderState.Incompatible) != 0)
            {
                return Color.Gray;
            } else
            if ((state & RenderState.Compatible) != 0)
            {
                return Color.White;
            } else
            if ((state & RenderState.Connected) != 0)
            {
                if ((state & RenderState.Backward) != 0)
                    return Color.Maroon;

                return Color.Black;
            } else
                return Color.LightGray;
        }

        static GraphicsPath GetArrowLinePath(float x1, float y1, float x2, float y2, out float centerX, out float centerY, bool include_arrow, float extra_thickness = 0)
        {
            var newPoints = DefaultStyle.GetArrowLinePoints(x1, y1, x2, y2, out centerX, out centerY, extra_thickness);

            var path = new GraphicsPath(FillMode.Winding);
            path.AddLines(newPoints.ToArray());
            if (include_arrow)
                path.AddLines(DefaultStyle.GetArrowPoints(x2, y2, extra_thickness).ToArray());
            path.CloseFigure();
            return path;
        }

        public static void RenderOutputConnection(Graphics graphics, NodeConnector output, float x, float y, RenderState state)
        {
            if (graphics == null ||
                output == null)
                return;
            
            RectangleF outputBounds;
            if (output.Node.Collapsed)	outputBounds = output.Node.outputBounds;
            else						outputBounds = output.bounds;

            var x1 = (outputBounds.Left + outputBounds.Right) / 2.0f;
            var y1 = (outputBounds.Top + outputBounds.Bottom) / 2.0f;
            
            float centerX;
            float centerY;
            using (var path = GetArrowLinePath(x1, y1, x, y, out centerX, out centerY, true, 0.0f))
            {
                using (var brush = new SolidBrush(GetArrowLineColor(state)))
                {
                    graphics.FillPath(brush, path);
                }
            }
        }
        
        public static void RenderInputConnection(Graphics graphics, NodeConnector input, float x, float y, RenderState state)
        {
            if (graphics == null || 
                input == null)
                return;
            
            RectangleF inputBounds;
            if (input.Node.Collapsed)	inputBounds = input.Node.inputBounds;
            else						inputBounds = input.bounds;

            var x2 = (inputBounds.Left + inputBounds.Right) / 2.0f;
            var y2 = (inputBounds.Top + inputBounds.Bottom) / 2.0f;

            float centerX;
            float centerY;
            using (var path = GetArrowLinePath(x, y, x2, y2, out centerX, out centerY, true, 0.0f))
            {
                using (var brush = new SolidBrush(GetArrowLineColor(state)))
                {
                    graphics.FillPath(brush, path);
                }
            }
        }

        public static GraphicsPath CreateRoundedRectangle(SizeF size, PointF location)
        {
            int cornerSize			= (int)GraphConstants.CornerSize * 2;
            int connectorSize		= (int)GraphConstants.ConnectorSize;
            int halfConnectorSize	= (int)Math.Ceiling(connectorSize / 2.0f);

            var height				= size.Height;
            var width				= size.Width;
            var halfWidth			= width / 2.0f;
            var halfHeight			= height / 2.0f;
            var connectorOffset		= (int)Math.Floor((GraphConstants.MinimumItemHeight - GraphConstants.ConnectorSize) / 2.0f);
            var left				= location.X;
            var top					= location.Y;
            var right				= location.X + width;
            var bottom				= location.Y + height;

            var path = new GraphicsPath(FillMode.Winding);
            path.AddArc(left, top, cornerSize, cornerSize, 180, 90);
            path.AddArc(right - cornerSize, top, cornerSize, cornerSize, 270, 90);

            path.AddArc(right - cornerSize, bottom - cornerSize, cornerSize, cornerSize, 0, 90);
            path.AddArc(left, bottom - cornerSize, cornerSize, cornerSize, 90, 90);
            path.CloseFigure();
            return path;
        }
    }
}
