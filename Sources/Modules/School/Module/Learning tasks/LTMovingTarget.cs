using GoodAI.Modules.School.Common;
using GoodAI.Modules.School.Worlds;
using System;
using System.ComponentModel;
using System.Drawing;

namespace GoodAI.Modules.School.LearningTasks
{
    /// <author>GoodAI</author>
    /// <meta>Os</meta>
    /// <status>WIP</status>
    /// <summary>"Moving Target" learning task</summary>
    /// <description>
    /// Ability Name: Efficient navigation in a simple environment + moving target. The class is derived from LTApproach with some additions for the moving target (currently: circular movement).
    /// Current behaviour of the target, the target moves itself towards the trajectory (the ellipse contained inside the rectangle (the screen)) and then it follows the trajectory
    /// </description>
    [DisplayName("Reach moving target")]
    public class LTMovingTarget : LTApproach                           // Deriving from LTApproach
    {
        private float m_angle;                                                    // Used for the trajectory of the circular movement of the target
        private PointF m_trajectory;

        private int m_stepsTakenForOneCircle;                                     // How many steps will denote one complete cycle of the ellipse

        private readonly TSHintAttribute MOVING_VELOCITY = new TSHintAttribute("Moving Velocity", "", typeof(float), 0, 1);                 // The velocity of the Target when it's trying to avoid the agent
        private readonly TSHintAttribute ELLIPSE_SIZE = new TSHintAttribute("Ellipse Rectangle Ratio", "", typeof(float), 0, 1);            // The size of the ellipse inside the screen it ranges from 0 to 1: 1.0f is the biggest ellipse that can fit in the POW screen
        private readonly TSHintAttribute STEPS_TAKEN_FOR_ONE_CIRCLE = new TSHintAttribute("Steps for ellipse", "", typeof(float), 0, 1);    // How many steps it takes for one complete cycle of the ellipse
        private readonly TSHintAttribute AVOIDING_AGENT = new TSHintAttribute("If avoiding agent or not", "", typeof(float), 0, 1);         // Either 0 or 1, If it's 1, the target tries to move away from the agent instead of moving ina  circular way

        public LTMovingTarget() : this(null) { }

        public LTMovingTarget(SchoolWorld w)
            : base(w)
        {
            TSHints[TSHintAttributes.DEGREES_OF_FREEDOM] = 2;                   // Set degrees of freedom to 2: move in 4 directions (1 means move only right-left)
            //TSHints[TSHintAttributes.MAX_TARGET_DISTANCE] = 0.3f;

            TSHints[DISTANCE_BONUS_COEFFICENT] = 15f;                            // Coefficent of 4 means that the available steps to reach the target are "initialDistance (between agent and target) * 4" (for more info check LTApproach)

            //Reusing TSHints from LTApproach with some additions
            TSHints.Add(MOVING_VELOCITY, 1);
            TSHints.Add(ELLIPSE_SIZE, 0.55f);
            TSHints.Add(STEPS_TAKEN_FOR_ONE_CIRCLE, 600);
            TSHints.Add(AVOIDING_AGENT, 0);

            TSProgression.Clear();                                              // Clearing TSProgression that was declared in LTApproach before filling custom one

            TSProgression.Add(TSHints.Clone());

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.55f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 500 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.6f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 400 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.7f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 300 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.7f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 200 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { ELLIPSE_SIZE, 0.8f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 100 }
            });

            TSProgression.Add(
                new TrainingSetHints {
                    { MOVING_VELOCITY, 2 },
                    { ELLIPSE_SIZE, 0.65f },
                    { STEPS_TAKEN_FOR_ONE_CIRCLE, 1000 },
                    { AVOIDING_AGENT, 1 },
                    { DISTANCE_BONUS_COEFFICENT, 7f },
                    { TSHintAttributes.IMAGE_TEXTURE_BACKGROUND, 1f}
            });
        }

        public override void ExecuteStep()                                  // UpdateState calls base's equivalent and then its own additional functions
        {
            m_stepsTakenForOneCircle = (int)TSHints[STEPS_TAKEN_FOR_ONE_CIRCLE];

            // The movement of the moving target is circular, and this circular movement is represented by an ellipse inside a rectangle (the Screen)

            m_angle += 1f / m_stepsTakenForOneCircle;                  // Increase the m_angle (which will produce the coordinates that will be used to generate the point to move towards to)

            if (m_angle > 1f)
            {
                m_angle = 0;
            }

            m_trajectory = GetPointInEllipse(m_angle, TSHints[ELLIPSE_SIZE]);             // Find the current coordinates to follow

            if ((int)TSHints[AVOIDING_AGENT] == 0)
            {
                m_target.Position.X = m_trajectory.X;
                m_target.Position.Y = m_trajectory.Y;
            }

            // MoveTowardsPoint(m_trajectory.X, m_trajectory.Y, movingVelocity);    // Move target towards the desired m_trajectory

            if ((int)TSHints[AVOIDING_AGENT] == 1)                              // If avoidingAgent is required from the TSHints, compute additional moving step trying to avoid the agent
            {
                MoveAwayFromAgentStep((int)TSHints[MOVING_VELOCITY]);
            }
        }

        public override void CreateTarget()
        {
            //MyLog.INFO.WriteLine(" called CreateTarget() from LTMovingTarget");

            m_target = WrappedWorld.CreateTarget(new Point(0, 0));

            m_target.Size.Width = 30;
            m_target.Size.Height = 30;

            float newNumber = (float)m_rndGen.NextDouble();

            PointF randomPoint = GetPointInEllipse(newNumber, TSHints[ELLIPSE_SIZE]);
            m_angle = newNumber;                                                  // Adapt the new current m_angle to the point generated

            m_target.Position.X = randomPoint.X;
            m_target.Position.Y = randomPoint.Y;

            WrappedWorld.AddGameObject(m_target);
        }

        // The ellipse's coordinates are calculated using a canonical form in polar coordinates
        // m_angle varies from 0 to 360 degrees, but it's in radians, the range is from 0 to 1 (2Pi)
        // For details: https://en.wikipedia.org/wiki/Ellipse#Parametric_form_in_canonical_position

        public PointF GetPointInEllipse(float angle, float EllipseRatioSize)                     // m_angle ranges from 0 to 1 (2Pi), EllipseRatioSize ranges from 0 to 1  (1 being the biggest ellipse inside the screen)
        {
            float width = WrappedWorld.Viewport.Width * EllipseRatioSize;                            // Width of the rectangle that will contain the ellipse
            float height = WrappedWorld.Viewport.Height * EllipseRatioSize;                          // Height of the rectangle that will contain the ellipse

            float Cx = WrappedWorld.Scene.Width / 2 - m_agent.Size.Width / 2;                      // X coordinate of the center of the rectangle
            float Cy = WrappedWorld.Scene.Height / 2 - m_agent.Size.Height / 2;                    // Y coordinate of the center of the rectangle

            PointF returnResult = new PointF();

            returnResult.X = (float)(Cx + (width / 2) * Math.Cos(angle * (Math.PI * 2)));         // Get the x coordinate of a point in the trajectory represented as an ellipse (with variable m_angle)
            returnResult.Y = (float)(Cy + (height / 2) * Math.Sin(angle * (Math.PI * 2)));        // Get the y coordinate of a point in the trajectory represented as an ellipse (with variable m_angle)

            returnResult.X -= (m_target.Size.Width / 2);
            returnResult.Y -= (m_target.Size.Height / 2);

            return returnResult;
        }

        public void MoveAwayFromAgentStep(float velocity)
        {
            // At each step, the target can move only on 1 axis, so choose the axis depending on the distances
            if (Math.Abs(m_agent.Position.X - m_target.Position.X) > Math.Abs(m_agent.Position.Y - m_target.Position.Y))    // Check what axis has the smallest difference between 2 points, the movement can happen only by 1 axis at a time
            {
                if (m_target.Position.X < m_agent.Position.X)
                {
                    m_target.Position.X -= velocity;
                }
                else if (m_target.Position.X > m_agent.Position.X)
                {
                    m_target.Position.X += velocity;
                }
            }
            else                                                        // Move on Y axis
            {
                if (m_target.Position.Y < m_agent.Position.Y)
                {
                    m_target.Position.Y -= velocity;
                }
                else if (m_target.Position.Y > m_agent.Position.Y)
                {
                    m_target.Position.Y += velocity;
                }
            }
        }
    }
}
