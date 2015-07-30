using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.Serialization;

namespace Touchless.Vision.Contracts
{
    [DataContract]
    public class Frame
    {
        [DataMember]
        public int Id { get; set; }

        [IgnoreDataMember]
        public Bitmap OriginalImage { get; set; }

        private Bitmap _image;
        [IgnoreDataMember]
        public Bitmap Image
        {
            get
            {
                if (_image == null)
                {
                    _image = OriginalImage.Clone() as Bitmap;
                }

                return _image;
            }
            set { _image = value;}
        }

        public Frame(Bitmap originalImage)
        {
            Id = NextId();
            OriginalImage = originalImage;
        }

        [DataMember]
        public byte[] ImageData
        {
            get
            {
                byte[] data = null;

                if (Image != null)
                {
                    var memoryStream = new MemoryStream();
                    Image.Save(memoryStream, ImageFormat.Png);
                    memoryStream.Flush();
                    data = memoryStream.ToArray();
                    memoryStream.Close();
                }

                return data;
            }
            //Setter is only here for serialization purposes
            set { }
        }


        private static readonly object SyncObject = new object();
        private static int _nextId = 1;
        private static int NextId()
        {
            int result;
            lock (SyncObject)
            {
                result = _nextId;
                _nextId++;
            }

            return result;
        }
    }
}