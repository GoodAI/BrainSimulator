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
        protected GameObject m_question;
        private int m_pickIdx;
        private int m_stepCount;

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
                { TSHintAttributes.MAX_NUMBER_OF_ATTEMPTS, 10000 },
                { TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 2 },
                { TSHintAttributes.DEGREES_OF_FREEDOM, 1 },
                { TSHintAttributes.IMAGE_NOISE, 0 },
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
                { TSHintAttributes.RANDOMNESS_LEVEL, 1 },
            };

            base.SetHints(TSHints);


            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_ROTATION, 1);
            TSProgression.Add(new TrainingSetHints{
                                {TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 3},
                                {TSHintAttributes.DEGREES_OF_FREEDOM, 2}});

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_SIZE, 1);
            TSProgression.Add(TSHintAttributes.IMAGE_NOISE, 1);

            TSProgression.Add(IS_VARIABLE_DISTANCE, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 4);

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_POSITION, 1);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.2f);

            TSProgression.Add(TSHintAttributes.IS_VARIABLE_COLOR, 1);
            TSProgression.Add(TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS, 5);
            TSProgression.Add(TSHintAttributes.RANDOMNESS_LEVEL, 1.4f);
        }

        #endregion

        #region Init functions

        public override void PresentNewTrainingUnit()
        {
            m_stepCount = 0;

            // Scale the noise in the world base on randomness_level
            {
                float randomness = TSHints[TSHintAttributes.RANDOMNESS_LEVEL];
                WrappedWorld.ImageNoiseStandardDeviation = 7 * randomness * randomness;
            }

            int noObjects = (int)TSHints[TSHintAttributes.NUMBER_OF_DIFFERENT_OBJECTS];

            // Generate an artificial invisible agent
            m_agent = WrappedWorld.CreateNonVisibleAgent();
            Point agentPos = m_agent.GetGeometry().Location;
            m_agent.GameObjectStyle = GameObjectStyleType.None; // Prevent reseting movement vector when colliding with something from the top (default style is Platformer)

            // Generate shapes around the agent
            CreateTargets(noObjects, agentPos);

            // Pick one target and duplicate it in the pow center
            m_pickIdx = m_rndGen.Next(noObjects);
            var pick = m_targets[m_pickIdx];
            Color color = TSHints[TSHintAttributes.IS_VARIABLE_COLOR] > 0 ? LearningTaskHelpers.RandomVisibleColor(m_rndGen) : pick.maskColor;
            m_question = WrappedWorld.CreateShape(agentPos, (Shape.Shapes)m_shapeIdcs[m_pickIdx], color, GameObjectType.None, pick.Width, pick.Height);
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
                    distance *= 1 + 0.04f * LearningTaskHelpers.GetRandomGaussian(m_rndGen) * TSHints[TSHintAttributes.RANDOMNESS_LEVEL];

                Point pos = new Point((int)(Math.Cos(angle) * distance), (int)(Math.Sin(angle) * distance));


                // Determine shape size
                float scale = 1.4f;

                if (TSHints[TSHintAttributes.IS_VARIABLE_SIZE] > 0)
                    scale = scale + 0.2f * LearningTaskHelpers.GetRandomGaussian(m_rndGen) * TSHints[TSHintAttributes.RANDOMNESS_LEVEL];

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

        private void Resize<T>(ref T[] array, int count)
        {
            if (array == null || array.Length < count)
                array = new T[count];
        }

        #endregion

        #region Update functions

        protected override bool DidTrainingUnitComplete(ref bool wasUnitSuccessful)
        {
            // Don't let the agent wander around for too long
            if (m_stepCount++ > 35 / TSHints[TSHintAttributes.RANDOMNESS_LEVEL])
                return true;

            // Move the question shape with the invisible agent
            m_question.SetPosition(new Point(m_agent.X, m_agent.Y));


            foreach (var gameObject in m_targets)
            {
                if (m_agent.DistanceTo(gameObject) < 5)
                {
                    wasUnitSuccessful = gameObject == m_targets[m_pickIdx];
                    return true;
                }
            }

            return false;
        }

        #endregion
    }
}
