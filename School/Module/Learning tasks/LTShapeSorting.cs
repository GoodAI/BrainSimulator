using System;
using System.Collections.Generic;
using System.Drawing;
using GoodAI.Core.Utils;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using GoodAI.Modules.VSA;
using OpenTK;


namespace GoodAI.Modules.School.LearningTasks
{
    /// <meta>mm</meta>
    /// <summary>
    /// 
    /// </summary>
    public class LTShapeSorting : AbstractLearningTask<RoguelikeWorld>
    {
        #region Fields

        private readonly TSHintAttribute ERROR_TOLERANCE = new TSHintAttribute("Tolerance in rads", "", typeof(float), 0, 1); //check needed;
        private readonly TSHintAttribute IS_VARIABLE_DISTANCE = new TSHintAttribute("Is fixed distance to target?", "", typeof(bool), 0, 1); //check needed;

        protected readonly Random m_rndGen = new Random();
        protected MovableGameObject m_agent;
        protected GameObject[] m_targets;
        private int pickIdx;

        private readonly HashSet<float> m_candidates = new HashSet<float>();
        private float[] m_shapeIdcs;

        #endregion

        #region Initialization

        public LTShapeSorting()
        { }

        public LTShapeSorting(SchoolWorld w)
            : base(w)
        {
            TSHints = new TrainingSetHints
            {
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { TSHintAttributes.RANDOMNESS_LEVEL, 1 },
                // Rotation about origin
                { TSHintAttributes.IS_VARIABLE_POSITION, 0},
                // Shape scaling
                { TSHintAttributes.IS_VARIABLE_SIZE, 0 },
                // Rotation about center of gravity (of each shape)
                { TSHintAttributes.IS_VARIABLE_ROTATION, 0},
                // Random color
                { TSHintAttributes.IS_VARIABLE_COLOR, 0},
                // Random distance from origin
                { IS_VARIABLE_DISTANCE, 0},
                { ERROR_TOLERANCE, 0.2f } // in rads/Pi to each side
            };

            base.SetHints(TSHints);


            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_ROTATION, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3);

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(ERROR_TOLERANCE, 0.1f);

            TSProgression.Add(IS_VARIABLE_DISTANCE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 4);

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.2f);

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 5);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.4f);
        }

        #endregion

        #region Init functions

        protected override void PresentNewTrainingUnit()
        {
            // Scale the noise in the world base on randomness_level
            {
                float randomness = TSHints[TSHintAttributes.RANDOMNESS_LEVEL];
                WrappedWorld.ImageNoiseStandardDeviation = 7 * randomness * randomness;
            }

            int noObjects = (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS];

            // Generate an artificial invisible agent
            m_agent = WrappedWorld.CreateNonVisibleAgent();
            Point agentPos = WrappedWorld.GetInitPosition();
            m_agent.SetPosition(agentPos);

            // Generate shapes around the agent
            CreateTargets(noObjects, agentPos);

            // Pick one target and duplicate it in the pow center
            pickIdx = m_rndGen.Next(noObjects);
            var pick = m_targets[pickIdx];
            Color color = TSHints[TSHintAttributes.IS_VARIABLE_COLOR] > 0 ? LearningTaskHelpers.RandomVisibleColor(m_rndGen) : pick.maskColor;
            WrappedWorld.CreateShape(agentPos, (Shape.Shapes)m_shapeIdcs[pickIdx], color, GameObjectType.None, pick.Width, pick.Height);
        }

        public virtual void CreateTargets(int noObjects, Point center)
        {
            Size centerSz = new Size(center); // Is Size because there is no overload for +(Point,Point)........


            // Pick random unique shapes from the available pool
            Resize(ref m_shapeIdcs, noObjects);
            int shapeCount = Enum.GetValues(typeof(Shape.Shapes)).Length; // noObjects must be at most shapeCount
            MyCombinationBase.GenerateCombinationUnique(new ArraySegment<float>(m_shapeIdcs), m_candidates, 0, shapeCount, m_rndGen);


            // Setup initial shape's positional parameters
            float step = (float)(2 * Math.PI / noObjects); // rads
            float angle = TSHints[TSHintAttributes.IS_VARIABLE_POSITION] > 0
                ? (float)(m_rndGen.NextDouble() * step) : 0;
            float distance = Math.Min(WrappedWorld.POW_HEIGHT, WrappedWorld.POW_WIDTH) / 3f;

            Resize(ref m_targets, noObjects);

            for (int i = 0; i < m_targets.Length; i++)
            {
                // Determine shape position
                if (TSHints[IS_VARIABLE_DISTANCE] > 0)
                    distance += GetRandomGaussian() * TSHints[TSHintAttributes.RANDOMNESS_LEVEL];

                Point pos = new Point((int)(Math.Cos(angle) * distance), (int)(Math.Sin(angle) * distance));


                // Determine shape size
                float scale = 1.4f;

                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] > 0)
                    scale = scale + 0.2f * GetRandomGaussian() * TSHints[TSHintAttributes.RANDOMNESS_LEVEL];

                Size size = new Size((int)(16 * scale), (int)(16 * scale));


                // Determine shape rotation
                float rotation = 0;

                if (TSHints[TSHintAttributes.IS_VARIABLE_ROTATION] > 0)
                    rotation = (float)(m_rndGen.Next() * 2 * Math.PI);


                // Determine shape color
                Color color = Color.White;

                if (TSHints[TSHintAttributes.IS_VARIABLE_COLOR] > 0)
                    color = LearningTaskHelpers.RandomVisibleColor(m_rndGen);


                // Create the correct shape
                m_targets[i] = WrappedWorld.CreateShape(pos + centerSz, (Shape.Shapes)m_shapeIdcs[i], color, size);
                m_targets[i].Rotation = rotation;


                angle += step;
            }
        }

        protected float GetRandomGaussian()
        {
            float u1 = Convert.ToSingle(m_rndGen.NextDouble()); //these are uniform(0,1) random doubles
            float u2 = Convert.ToSingle(m_rndGen.NextDouble()); //these are uniform(0,1) random doubles
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2)); //random normal(0,1)
            return randStdNormal;
        }

        private void Resize<T>(ref T[] array, int count)
        {
            if (array == null || array.Length < count)
                array = new T[count];
        }

        #endregion

        #region Update functions

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            Vector2 origin = new Vector2(m_agent.previousX, m_agent.previousY);

            Vector2 agentDir = new Vector2(m_agent.vX, m_agent.vY);
            agentDir.NormalizeFast();

            var pick = m_targets[pickIdx];
            Vector2 pickDir = new Vector2(pick.X, pick.Y) - origin;
            pickDir.NormalizeFast();

            float cos;
            Vector2.Dot(ref agentDir, ref pickDir, out cos);

            //if (1 - cos < TSHints[ERROR_TOLERANCE])
            wasUnitSuccessful = true;

            return true;
        }

        #endregion
    }
}