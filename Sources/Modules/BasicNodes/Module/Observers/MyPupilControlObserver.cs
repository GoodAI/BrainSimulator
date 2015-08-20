using GoodAI.Core;
using GoodAI.Core.Observers;
using GoodAI.Core.Observers.Helper;
using GoodAI.Modules.Retina;
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.Observers
{
    public class MyPupilControlObserver : MyNodeObserver<MyPupilControl>
    {
        public MyPupilControlObserver()
        {
            m_kernel = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, 
                @"Observers\FocuserInputObserver", "PupilControlObserver");                     
        }

        private CudaDeviceVariable<float> m_StringDeviceBuffer;

        protected override void Execute()
        {
            m_kernel.SetupExecution(Target.AttentionMap.Count);
            m_kernel.Run(Target.AttentionMap, Target.Centroids, Target.CentroidsCount, TextureWidth, TextureHeight, VBODevicePointer);

            for (int i = 0; i < Target.CentroidsCount; i++)
            {
                int x = (int)((Target.Centroids.Host[i * MyPupilControl.CENTROID_FIELDS] + 1) * 0.5f * TextureWidth);
                int y = (int)((Target.Centroids.Host[i * MyPupilControl.CENTROID_FIELDS + 1] + 1) * 0.5f * TextureHeight);
                float DBI = Target.Centroids.Host[i * MyPupilControl.CENTROID_FIELDS + 5];
                MyDrawStringHelper.String2Index(i + " ", m_StringDeviceBuffer);
                MyDrawStringHelper.DrawStringFromGPUMem(m_StringDeviceBuffer, x - 4, y - 14, 0, 0xFF69A5FF, VBODevicePointer, TextureWidth, TextureHeight,0,(i+" ").Length);
                //MyDrawStringHelper.DrawDecimalString(DBI.ToString("0.0000") , x - 10, y + 2, 0, 0xFF69A5FF, VBODevicePointer, TextureWidth, TextureHeight);
            }
        }
        
        protected override void Reset()
        {
            TextureWidth = Target.AttentionMap.ColumnHint;
            TextureHeight = Target.AttentionMap.Count / TextureWidth;            
        }
    }
}
