using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;

/*
 * Mozna nepotrebne?
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
 */


namespace XmlFeedForwardNet.Layers
{
    /*
     * Manually Specify Width, Height, Nb.
     * Size and Count recalculation are triggered by the setters.
     */

    public struct MyLayerDim
    {
        // GPU pointer to the offseted data
        public CUdeviceptr Ptr;    


        // Number of images
        public SizeT Nb
        {
            get { return m_nb; }
            set { m_nb = value; m_count = m_nb * m_size; }
        }


        // Width of each image
        public SizeT Width
        {
            get { return m_width; }
            set { m_width = value; Size = m_width * m_height * m_depth; }
        }

        // Height of each image
        public SizeT Height
        {
            get { return m_height; }
            set { m_height = value; Size = m_width * m_height * m_depth; }
        }

        // Depth of each image
        public SizeT Depth
        {
            get { return m_depth; }
            set { m_depth = value; Size = m_width * m_height * m_depth; }
        }

        // Size of each image (Width * Height * Depth)
        public SizeT Size
        {
            get { return m_size; }
            private set { m_size = value; m_count = m_nb * m_size; }
        }

        // Total number of channels (Nb * Size)
        public SizeT Count
        {
            get { return m_count; }
        }

        public static uint GetStructSize()
        {
            if (m_structSize == 0)
            {
                MyLayerDim dummy = new MyLayerDim();
                m_structSize = (uint)Marshal.SizeOf(dummy);
            }
            return m_structSize;
        }

        private SizeT m_nb;
        private SizeT m_width;
        private SizeT m_height;
        private SizeT m_depth;
        private SizeT m_size;
        private SizeT m_count;
        private static uint m_structSize = 0;


        public override string ToString()
        {
            return Nb + "x[" + Width + "x" + Height + "x" + Depth + "]";
        }
    }
}
