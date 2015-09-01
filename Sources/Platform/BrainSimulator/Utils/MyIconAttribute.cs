using System;
using System.Drawing;

namespace GoodAI.BrainSimulator.Utils
{
    [AttributeUsage(AttributeTargets.Class)]
    class MyIconAttribute : Attribute
    {
        public string Big { get; set; }
        public string Small { get; set; }

        private Bitmap GetBitmap(string name)
        {
            return (Bitmap)Properties.Resources.ResourceManager.GetObject(name, Properties.Resources.Culture);            
        }

        public Bitmap BigIcon { get { return GetBitmap(Big); } }
        public Bitmap SmallIcon { get { return GetBitmap(Small); } }
    }    
}
