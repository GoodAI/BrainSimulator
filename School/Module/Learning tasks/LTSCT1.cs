using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.School.LearningTasks
{
    [DisplayName("LTSCT1 - 1 shape")]
    public class Ltsct1 : AbstractLearningTask<RoguelikeWorld>
    {
        private readonly Random m_rndGen = new Random();
        protected bool[][] generationsCheckTable;
        protected ScFixPositions m_positions;
        protected ScFixColors m_colors;
        private FileStream m_fs;

        public Ltsct1() : this(null) { }

        public Ltsct1(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints {
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 1000000 },
                { TSHintAttributes.IMAGE_NOISE, 1}
            };

            TSProgression.Add(TSHints.Clone());
        }

        public virtual void InitCheckTable()
        {
            generationsCheckTable = new bool[ScConstants.numPositions][];
            for (int i = 0; i < generationsCheckTable.Length; i++)
            {
                generationsCheckTable[i] = new bool[ScConstants.numShapes];
            }

            m_fs = new FileStream(@"D:\summerCampSamples\SCT1.csv", FileMode.Append);
        }

        public override void Init()
        {
            base.Init();
            InitCheckTable();
            m_positions = new ScFixPositions(WrappedWorld.GetPowGeometry());
            m_colors = new ScFixColors(ScConstants.numColors, WrappedWorld.BackgroundColor);
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

        private int m_shapeIndex;
        protected virtual bool CheckTableFull()
        {
            return !generationsCheckTable.Any(b => b.Any(b1 => b1 == false));
        }

        protected virtual void CreateScene()
        {
            if (m_rndGen.Next(ScConstants.numShapes + 1) == 1) return; // no shape, no target

            int randomLocationIdx = m_rndGen.Next(ScConstants.numPositions);

            AddShape(randomLocationIdx);
        }

        protected virtual void AddShape(int randomLocationIndex)
        {
            SizeF size = new SizeF(WrappedWorld.GetPowGeometry().Width/4, WrappedWorld.GetPowGeometry().Height/4);

            Color color = m_colors.GetRandomColor(m_rndGen);

            PointF location = m_positions.Positions[randomLocationIndex];

            m_shapeIndex = m_rndGen.Next(ScConstants.numShapes);
            Shape.Shapes randomShape = (Shape.Shapes)m_shapeIndex;

            WrappedWorld.CreateShape(randomShape, color, location, size);

            generationsCheckTable[randomLocationIndex][m_shapeIndex] = true;
        }
    }
}
