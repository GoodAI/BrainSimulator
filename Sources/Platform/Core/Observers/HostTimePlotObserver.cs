using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Utils;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace GoodAI.Platform.Core.Observers
{
    public class HostTimePlotObserver : MyObserver<MyMemoryBlock<float>>
    {
        public HostTimePlotObserver()
        {
            TextureWidth = 800;
            TextureHeight = 400;
        }

        protected override void Execute()
        {
            Target.SafeCopyToHost();

            Bitmap bmp = new Bitmap(@"C:\test.bmp");
            BitmapData bmp_data = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            GL.BindTexture(TextureTarget.Texture2D, TextureId);
 
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, bmp_data.Width, bmp_data.Height, 0,
                OpenTK.Graphics.OpenGL.PixelFormat.Bgra, PixelType.UnsignedByte, bmp_data.Scan0);

            bmp.UnlockBits(bmp_data);
        }

        public override string GetTargetName(MyNode declaredOwner)
        {
            if (declaredOwner == Target.Owner)
            {
                return Target.Owner.Name + ": " + Target.Name;
            }
            else
            {
                return declaredOwner.Name + " (" + Target.Owner.Name + "): " + Target.Name;
            }
        }

        protected override string CreateTargetIdentifier()
        {
            if (Target != null)
            {
                return Target.Owner.Id + "#" + Target.Name;
            }
            else return String.Empty;
        }

        public override void RestoreTargetFromIdentifier(MyProject project)
        {
            if (TargetIdentifier != null)
            {
                string[] split = TargetIdentifier.Split('#');
                if (split.Length == 2)
                {
                    MyWorkingNode node = (MyWorkingNode)project.GetNodeById(int.Parse(split[0]));

                    if (node != null)
                    {
                        Target = (MyMemoryBlock<float>)MyMemoryManager.Instance.GetMemoryBlockByName(node, split[1]);
                    }
                }
            }
        }
    }
}
