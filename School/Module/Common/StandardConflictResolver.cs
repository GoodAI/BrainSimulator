using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.School.Common
{
    public class StandardConflictResolver
    {
        public void Resolve(MovableGameObject mgo1, GameObject o2)
        {

            MovableGameObject mgo2 = o2 as MovableGameObject;

            if (mgo1.Subtype != null || o2.Subtype != null)
            {
                if (ResolveSubtype(mgo1, o2))
                {
                    return;
                }
            }


            // both objects are movable
            if (mgo2 != null)
            {
                if (mgo2.type == GameObjectType.Enemy)
                {
                    // TODO
                }
                return;
            }


            else // only one object is movable
            {
                // other object interactions
                ISwitchable switcher = o2 as ISwitchable;
                if (switcher != null && switcher.SwitchOnCollision())
                {
                    switcher.Switch();
                }

                if (!(
                    o2.type == GameObjectType.Agent ||
                    o2.type == GameObjectType.ClosedDoor ||
                    o2.type == GameObjectType.Obstacle ||
                    o2.type == GameObjectType.Teacher ||
                    o2.type == GameObjectType.None))
                {
                    return;
                }

                if (mgo1.GameObjectStyle == GameObjectStyleType.Platformer)   // Collision outcome depends on the GameObjectStyleType
                {
                    SetPlattformerReaction(mgo1, o2);
                }
                else if (mgo1.GameObjectStyle == GameObjectStyleType.Pinball)
                {
                    SetPinballReaction(mgo1, o2);
                }
                else if (mgo1.GameObjectStyle == GameObjectStyleType.None)
                {
                }

                return;
            }


            throw new ArgumentException("For given objects doesn't exist confict resolution. " +
            "Did you forget to implement overriding class for StandardConflictResolver?");
        }

        // returns false if subtype combine its behavior with main type
        // true if overrides main type or have no main type
        public virtual bool ResolveSubtype(GameObject o1, GameObject o2)
        {
            return false;
        }

        public void SetPlattformerReaction(MovableGameObject mgo1, GameObject o2)
        {
            int collideResult = CheckCollision(mgo1, o2);
            Point lastUntouchingPosition = GetLastUntouchingPosition(mgo1, o2, ref collideResult); //finds the X,Y position of "source" in the closest point to "o2" before collision)

            if (collideResult == 4 || collideResult == 1)               // If it collided at the bottom or top
            {
                mgo1.Y = lastUntouchingPosition.Y;                    // Reposition Y position to the last position where there was no collision
                mgo1.vY = 0;                                          // and reset Y velocity
            }
            if (collideResult == 2 || collideResult == 3)               // If it collided at the right or left reset X velocity to 0
            {
                mgo1.X = lastUntouchingPosition.X;                    // Reposition X position to the last position where there was no collision
                mgo1.vX = 0;                                          // and reset X velocity
            }

            if (mgo1.previousY == mgo1.Y && collideResult == 4)     // Remember if object is on ground, collision on the bottom
            {
                mgo1.onGround = true;
            }

        }

        public void SetPinballReaction(MovableGameObject mgo1, GameObject o2)
        {
            int collideResult = 0;
            Point lastUntouchingPosition = GetLastUntouchingPosition(mgo1, o2, ref collideResult); //finds the X,Y position of "source" in the closest point to "o2" before collision)
            if (collideResult == 4 || collideResult == 1)               // If it collided at the bottom or top, reposition and invert Y velocity
            {
                mgo1.Y = lastUntouchingPosition.Y;
                mgo1.vY = -mgo1.vY;
            }
            if (collideResult == 2 || collideResult == 3)               // If it collided at the right or left, reposition and invert X velocity
            {
                mgo1.X = lastUntouchingPosition.X;
                mgo1.vX = -mgo1.vX;
            }
            if (mgo1.IsAffectedByGravity == true)                     //If it's affected by gravity, add deceleration effect
            {
                mgo1.vX *= 0.93f;                                     //TODO: replace 0.93f constant with a variable
                mgo1.vY *= 0.93f;
            }

        }

        public static Point ReturnCoordinatesBetweenTwoPoints(int SourceX, int SourceY, int TargetX, int TargetY, float blend)
        {
            Point result = new Point(); ;
            result.X = (int)(SourceX + blend * (TargetX - SourceX));
            result.Y = (int)(SourceY + blend * (TargetY - SourceY));
            return result;
        }

        /*
                    * This method is called when 2 GameObjects collide, it finds the point of contact by applying a binary search where the lower side is Source's X,Y previous position and the higher side is
                    * Source's X,Y current position, and the middle (needed by the binary search) is found by returning the middle position between the 2 points.
                    */
        public Point GetLastUntouchingPosition(MovableGameObject source, GameObject target, ref int lastCollideResult)
        {
            int currentCollisionResult = 0;                                             // The collision result used while iteratively repositioning the Source
            Point lowerSide = new Point(source.previousX, source.previousY);              // Initialise the lower side for the binary search to the coordinates of the source's previous position
            Point higherSide = new Point(source.X, source.Y);                             // Initialise the higher side for the binary search to the coordinates of the source's current position
            Point currentMiddle = new Point(-10, -10);                                    // Declares the middle point, which will be used for the binary search
            Point previousMiddle = new Point(0, 0);                                       // Declares the previous middle point, which will be used to decide when the binary search can't divide anymore
            Point originalSourcePosition = new Point(source.X, source.Y);
            Point lastUntouchingPosition = new Point(source.previousX, source.previousY); //It will be used to remember the last position encountered where there was no contact (collision) in order to apply repositioning
            int counter = 0;

            //Stop the loop when the function can't divide anymore by half the distance between lowerSide and higherSide
            while (previousMiddle != currentMiddle)                         // If the previous middle point and the current one are equivalent, it means that distance can't be divided anymore by half, exit
            {
                previousMiddle = currentMiddle;                             // Update the previous value  of the middle point, it is used to check if they are equivalent, if they are, the loop should stop
                currentMiddle = ReturnCoordinatesBetweenTwoPoints(lowerSide.X, lowerSide.Y, higherSide.X, higherSide.Y, 0.5f);  // Find the middle point between lowerSide and higherSide

                //MyLog.DEBUG.WriteLine("Recomputed middle between : " + lowerSide.x + "," + lowerSide.y + " | " + higherSide.x + "," + higherSide.y + " : = " + currentMiddle.x + ", " + currentMiddle.y);

                source.X = currentMiddle.X;                                 // Position source's x to the point in exam
                source.Y = currentMiddle.Y;                                 // Position source's y to the point in exam
                currentCollisionResult = CheckCollision(source, target);    // Check if there is a collision using the point in exam

                //MyLog.DEBUG.WriteLine("Collision result: " + currentCollisionResult);

                if (currentCollisionResult > 0)                             // If there is a collision reposition higherSide for the binary search
                {
                    higherSide = currentMiddle;
                    lastCollideResult = currentCollisionResult;
                }
                else                                                        // If there is no collision reposition the lowerSide for the binary search
                {
                    lowerSide = currentMiddle;
                    lastUntouchingPosition.X = currentMiddle.X;             // Update the last encountered position where there was no collision
                    lastUntouchingPosition.Y = currentMiddle.Y;
                }

                if (counter > 20)                                           // Additional check
                {
                    MyLog.ERROR.WriteLine("Too many iterations during repositioning, this should never happen, exiting..");
                    break;
                }

                counter++;
            }

            source.X = originalSourcePosition.X;
            source.Y = originalSourcePosition.Y;

            return lastUntouchingPosition;
        }

        /*public bool CheckCollision(GameObject o1, GameObject o2)
        {
            
        }*/

        public static int CheckCollision(GameObject SourceGameObject, GameObject TargetGameObject)
        {
            float w = 0.5f * (SourceGameObject.Width + TargetGameObject.Width);
            float h = 0.5f * (SourceGameObject.Height + TargetGameObject.Height);

            float dx = (SourceGameObject.X + (SourceGameObject.Width / 2)) - (TargetGameObject.X + (TargetGameObject.Width / 2));
            float dy = (SourceGameObject.Y + (SourceGameObject.Height / 2)) - (TargetGameObject.Y + (TargetGameObject.Height / 2));

            MovableGameObject mobj = SourceGameObject as MovableGameObject;

            if (Math.Abs(dx) <= 0.99f * w && Math.Abs(dy) <= 0.99f * h) // 0.99 = make it more float-robust
            {
                //If we reached this point, there is a collision, so check the side of collision
                float wy = w * dy;
                float hx = h * dx;

                if (wy > hx)
                {
                    if (wy < -hx)
                    {
                        if (mobj.vX < 0)  //If you were moving on the left the collision can't be on the right side, return 1(Top)
                        {
                            return 1;
                        }
                        return 2;       //Collision on the right
                    }
                    else
                    {
                        return 1;       //Collision at the top
                    }
                }
                else
                {
                    if (wy > -hx)
                    {
                        if (mobj.vX > 0)  //If you were moving on the right the collision can't be on the left side, return 4(Bottom)
                        {
                            return 4;
                        }
                        return 3;       //Collision on the left
                    }
                    else
                    {
                        return 4;       //Collision at the bottom
                    }
                }

            }

            //Otherwise, there was no collision
            return 0;
        }
    }

    //public class SampleConflictResolver : StandardConflictResolver
    //{
    //    
    //    public override bool ResolveSubtype(GameObject o1, GameObject o2)
    //    {
    //        return false;
    //    }
    //}
}
