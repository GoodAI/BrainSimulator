using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using GoodAI.Core.Utils;
using GoodAI.School.Learning_tasks;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT1 - 1 shape")]
    public class Ltsct1 : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly Random m_rndGen = new Random();
        protected bool[][] GenerationsCheckTable;
        protected ScFixPositions Positions;
        protected ScFixColors Colors;
        protected FileStream FileStream;

        public Ltsct1() : this(null) { }
        protected virtual string Path { get { return @"D:\summerCampSamples\SCT1.csv";} }

        public Ltsct1(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1},
                { TSHintAttributes.IMAGE_NOISE_BLACK_AND_WHITE, 1}
            };

            TSProgression.Add(TSHints.Clone());

            
        }

        public virtual void InitCheckTable()
        {
            GenerationsCheckTable = new bool[ScConstants.numPositions][];
            for (int i = 0; i < GenerationsCheckTable.Length; i++)
            {
                GenerationsCheckTable[i] = new bool[ScConstants.numShapes];
            }
        }

        public override void Init()
        {
            base.Init();
            InitCheckTable();
            Positions = new ScFixPositions(WrappedWorld.GetPowGeometry());
            Colors = new ScFixColors(ScConstants.numColors, WrappedWorld.BackgroundColor);
            WrappedWorld.ImageNoiseStandardDeviation = 256.0f / ScConstants.numColors / 2;

            OpenFileStream();
        }

        protected virtual void OpenFileStream()
        {
            var path = Path;
            if (!File.Exists(path))
            {
                File.Create(path);
            }
            if (FileStream != null)
            {
                FileStream.Dispose();
            }
            FileStream = new FileStream(path, FileMode.Truncate);
        }


        public override void PresentNewTrainingUnit()
        {
            WrappedWorld.CreateNonVisibleAgent();
            CreateScene();
        }

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            wasUnitSuccessful = false;

            if (CheckTableFull())
            {
                MyLog.INFO.WriteLine("Set Is Complete!");
                //wasUnitSuccessful = true;
            }

            return true;
        }

        protected virtual bool CheckTableFull()
        {
            return !GenerationsCheckTable.Any(b => b.Any(b1 => b1 == false));
        }

        protected AvatarsActions Actions;
        protected virtual void CreateScene()
        {
            Actions = new AvatarsActions();
            if (m_rndGen.Next(ScConstants.numShapes + 1) > 0)
            {
            {
                return; // no shape, no target
            }
                int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);
                AddShape(randomLocationIdx);
            }
            Actions.Shapes[m_shapeIndex] = true;
            
            Actions.WriteActions(FileStream);
            
        }

        private int m_shapeIndex;

        protected virtual void AddShape(int randomLocationIndex)
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

            Color color = Colors.GetRandomColor(m_rndGen);

            PointF location = Positions.Positions[randomLocationIndex];

            m_shapeIndex = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)m_shapeIndex;

            WrappedWorld.CreateShape(randomShape, color, location, size);

            GenerationsCheckTable[randomLocationIndex][m_shapeIndex] = true;
        }
    }
}
