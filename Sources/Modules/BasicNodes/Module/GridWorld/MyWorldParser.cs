using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;



namespace GoodAI.Modules.GridWorld
{
    public class MyMapI : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, A, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                              
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 0
            };
            m_width = 10;
            m_height = 10;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOff();
            this.PlaceObjectTo(new int2(5, 5), ll);

            LightsControl lc = new LightsControl();
            lc.myObjects.Add(ll);
            lc.TurnOff();
            this.PlaceObjectTo(new int2(1, 1), lc);

            Lights lll = new Lights();
            lll.TurnOn();
            this.PlaceObjectTo(new int2(2, 5), lll);

            LightsControl lcc = new LightsControl();
            lcc.myObjects.Add(lll);
            lcc.TurnOn();
            this.PlaceObjectTo(new int2(4, 7), lcc);

            MyDoor door = new MyDoor();
            door.TurnOff();
            this.PlaceObjectTo(new int2(5, 1), door);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door);
            dc.TurnOff();
            this.PlaceObjectTo(new int2(7, 7), dc);

            MyDoor door2 = new MyDoor();
            door2.TurnOn();
            this.PlaceObjectTo(new int2(7, 4), door2);

            DoorControl dc2 = new DoorControl();
            dc2.myObjects.Add(door);
            dc2.TurnOn();
            this.PlaceObjectTo(new int2(2, 8), dc2);

            StaticObjects = new MyStaticObject[] { ll, lc, lll, lcc, door2, dc2, door, dc };
        }
    }

    /// <summary>
    /// definitions of custom maps here
    /// </summary>
    /// 
    public class MyMapH : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, A, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0
            };
            m_width = 7;
            m_height = 7;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOff();
            this.PlaceObjectTo(new int2(5, 5), ll);
            /*
            Lights lll = new Lights();
            lll.TurnOff();
            this.PlaceObjectTo(new int2(2, 5), lll);

            Lights llll = new Lights();
            llll.TurnOff();
            this.PlaceObjectTo(new int2(2, 2), llll);

            */
            LightsControl lc = new LightsControl();
            lc.TurnOff();
            lc.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(1, 1), lc);

            MyDoor door2 = new MyDoor();
            door2.TurnOff();
            this.PlaceObjectTo(new int2(5, 1), door2);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door2);
            this.PlaceObjectTo(new int2(1, 5), dc);

            StaticObjects = new MyStaticObject[] { ll, lc, door2, dc };//,lll,llll};

        }
    }

    /// <summary>
    /// definitions of custom maps here
    /// </summary>
    /// 
    public class MyMapJ : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, A, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0
            };
            m_width = 7;
            m_height = 7;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOff();
            this.PlaceObjectTo(new int2(4, 4), ll);
            /*
            Lights lll = new Lights();
            lll.TurnOff();
            this.PlaceObjectTo(new int2(2, 5), lll);

            Lights llll = new Lights();
            llll.TurnOff();
            this.PlaceObjectTo(new int2(2, 2), llll);

            */
            LightsControl lc = new LightsControl();
            lc.TurnOff();
            lc.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(2, 2), lc);

            MyDoor door2 = new MyDoor();
            door2.TurnOff();
            this.PlaceObjectTo(new int2(4, 2), door2);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door2);
            this.PlaceObjectTo(new int2(2, 4), dc);

            StaticObjects = new MyStaticObject[] { ll, lc, door2, dc };//,lll,llll};

        }
    }


    /// <summary>
    /// definitions of custom maps here
    /// </summary>
    /// 
    public class MyMapL : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                1, 1, 0,
                0, 0, 0,
                0, 0, 0,
                1, 1, 0,
                1, 0, A,
            };
            m_width = 3;
            m_height = 5;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            MyDoor door2 = new MyDoor();
            door2.TurnOff();
            this.PlaceObjectTo(new int2(2, 3), door2);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door2);
            this.PlaceObjectTo(new int2(0, 2), dc);

            DoorControl dc2 = new DoorControl();
            dc2.myObjects.Add(door2);
            this.PlaceObjectTo(new int2(1, 4), dc2);


            StaticObjects = new MyStaticObject[] { door2, dc, dc2 };//,lll,llll};

        }
    }

    /// <summary>
    /// definitions of custom maps here
    /// </summary>
    /// 
    public class MyMapK : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, A, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            };
            m_width = 5;
            m_height = 5;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOff();
            this.PlaceObjectTo(new int2(3, 3), ll);
            /*
            Lights lll = new Lights();
            lll.TurnOff();
            this.PlaceObjectTo(new int2(2, 5), lll);

            Lights llll = new Lights();
            llll.TurnOff();
            this.PlaceObjectTo(new int2(2, 2), llll);

            */
            LightsControl lc = new LightsControl();
            lc.TurnOff();
            lc.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(1, 1), lc);

            MyDoor door2 = new MyDoor();
            door2.TurnOff();
            this.PlaceObjectTo(new int2(3, 1), door2);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door2);
            this.PlaceObjectTo(new int2(1, 3), dc);

            StaticObjects = new MyStaticObject[] { ll, lc, door2, dc };//,lll,llll};

        }
    }

    public class MyMapE : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 1, 
                0, 0, 0, 0, 0, 1, 
                0, 0, 1, 0, 0, 1,
                0, 0, 0, 0, 0, 1, 
                0, 0, A, 0, 0, 0, 
                0, 0, 0, 0, 0, 0
            };
            m_width = 6;
            m_height = 6;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOn();
            this.PlaceObjectTo(new int2(4, 4), ll);

            LightsControl lc = new LightsControl();
            lc.TurnOn();
            lc.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(1, 1), lc);

            LightsControl lc2 = new LightsControl();
            lc2.TurnOn();
            lc2.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(3, 3), lc2);

            LightsControl lc3 = new LightsControl();
            lc3.TurnOn();
            lc3.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(2, 2), lc3);


            StaticObjects = new MyStaticObject[4];
            StaticObjects[0] = ll;
            StaticObjects[1] = lc;
            StaticObjects[2] = lc2;
            StaticObjects[3] = lc3;
        }
    }

    public class MyMapB : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, A, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,                              
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0
            };
            m_width = 10;
            m_height = 10;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOn();
            this.PlaceObjectTo(new int2(7, 7), ll);

            LightsControl lc = new LightsControl();
            lc.TurnOn();
            lc.myObjects.Add(ll);
            this.PlaceObjectTo(new int2(1, 1), lc);

            StaticObjects = new MyStaticObject[2];
            StaticObjects[0] = ll;
            StaticObjects[1] = lc;
        }
    }

    public class MyMapC : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, A, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,                              
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0
            };
            m_width = 10;
            m_height = 10;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOn();
            this.PlaceObjectTo(new int2(7, 7), ll);

            LightsControl lc = new LightsControl();
            lc.myObjects.Add(ll);
            lc.TurnOn();
            this.PlaceObjectTo(new int2(1, 1), lc);

            /*
            // place into the map
            Door door = new Door();
            door.turnOff();
            this.placeObjectTo(new int2(5, 4), door);
            */
            MyDoor door2 = new MyDoor();
            door2.TurnOff();
            this.PlaceObjectTo(new int2(5, 5), door2);

            /*
            DoorControl dc = new DoorControl();
            //dc.myObjects.Add(door);
            dc.myObjects.Add(door2);
            this.PlaceObjectTo(new int2(1, 8), dc);
            */

            DoorControl dc2 = new DoorControl();
            //dc2.myObjects.Add(door);
            dc2.myObjects.Add(door2);
            dc2.TurnOff();
            this.PlaceObjectTo(new int2(8, 1), dc2);
            StaticObjects = new MyStaticObject[] { ll, lc, dc2, door2 };
        }
    }

    public class MyMapF : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, A, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,                              
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0
            };
            m_width = 10;
            m_height = 10;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOn();
            this.PlaceObjectTo(new int2(6, 0), ll);

            LightsControl lc = new LightsControl();
            lc.myObjects.Add(ll);
            lc.TurnOn();
            this.PlaceObjectTo(new int2(1, 1), lc);

            // lights control
            Lights llX = new Lights();
            llX.TurnOn();
            this.PlaceObjectTo(new int2(6, 9), llX);

            LightsControl lcX = new LightsControl();
            lcX.myObjects.Add(llX);
            lcX.TurnOn();
            this.PlaceObjectTo(new int2(9, 9), lcX);

            // place into the map
            MyDoor doorX = new MyDoor();
            doorX.TurnOff();
            this.PlaceObjectTo(new int2(8, 6), doorX);

            DoorControl doorControlX = new DoorControl();
            doorControlX.myObjects.Add(doorX);
            doorControlX.TurnOff();
            this.PlaceObjectTo(new int2(4, 7), doorControlX);

            // door 2 with two controls
            MyDoor door2 = new MyDoor();
            door2.TurnOff();
            this.PlaceObjectTo(new int2(5, 5), door2);

            DoorControl dc = new DoorControl();
            //dc.myObjects.Add(door);
            dc.myObjects.Add(door2);
            dc.TurnOff();
            this.PlaceObjectTo(new int2(1, 8), dc);

            DoorControl dc2 = new DoorControl();
            //dc2.myObjects.Add(door);
            dc2.myObjects.Add(door2);
            dc2.TurnOff();
            this.PlaceObjectTo(new int2(8, 1), dc2);

            StaticObjects = new MyStaticObject[] { dc, dc2, door2, doorX, doorControlX, lcX, llX, ll, lc };
        }
    }

    public class MyMapD : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, A, 0, 0,
                0, 0, 0, 0, 0,
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            };
            m_width = 5;
            m_height = 11;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            Lights ll = new Lights();
            ll.TurnOn();
            this.PlaceObjectTo(new int2(3, 9), ll);

            LightsControl lc = new LightsControl();
            lc.myObjects.Add(ll);
            lc.TurnOn();
            this.PlaceObjectTo(new int2(1, 1), lc);

            StaticObjects = new MyStaticObject[2];
            StaticObjects[0] = ll;
            StaticObjects[1] = lc;
        }
    }

    public class MyMapA : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, A, 0, 0,                               
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };
            m_width = 16;
            m_height = 10;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            // place into the map
            MyDoor door = new MyDoor();
            door.TurnOff();
            this.PlaceObjectTo(new int2(12, 5), door);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door);
            dc.TurnOff();
            this.PlaceObjectTo(new int2(9, 9), dc);

            DoorControl dc2 = new DoorControl();
            dc2.myObjects.Add(door);
            dc2.TurnOff();
            this.PlaceObjectTo(new int2(10, 1), dc2);

            Lights l = new Lights();
            l.TurnOff();
            this.PlaceObjectTo(new int2(5, 5), l);

            Lights ll = new Lights();
            ll.TurnOff();
            this.PlaceObjectTo(new int2(1, 1), ll);

            LightsControl lc = new LightsControl();
            lc.myObjects.Add(l);
            lc.myObjects.Add(ll);
            lc.TurnOff();
            this.PlaceObjectTo(new int2(6, 6), lc);

            StaticObjects = new MyStaticObject[] { dc, door, dc2, lc, l, ll };
        }
    }

    public class MyMapG : AbstractMap
    {
        protected override void DefineArray()
        {
            m_array = new int[] 
            {                 
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                               
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                A, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
            };
            m_width = 16;
            m_height = 10;
        }

        protected override void DefineMovingObjects()
        {
            MovingObjects = new MyMovingObject[0];
        }

        protected override void DefineStaticObjects()
        {
            // place into the map
            MyDoor door = new MyDoor();
            door.TurnOn();
            this.PlaceObjectTo(new int2(2, 3), door);

            DoorControl dc = new DoorControl();
            dc.myObjects.Add(door);
            dc.TurnOn();
            this.PlaceObjectTo(new int2(11, 7), dc);

            MyDoor door2 = new MyDoor();
            door2.TurnOn();
            this.PlaceObjectTo(new int2(7, 7), door2);

            DoorControl dc2 = new DoorControl();
            dc2.myObjects.Add(door2);
            dc2.TurnOn();
            this.PlaceObjectTo(new int2(12, 3), dc2);

            Lights l = new Lights();
            l.TurnOff();
            this.PlaceObjectTo(new int2(3, 7), l);

            LightsControl lc = new LightsControl();
            lc.myObjects.Add(l);
            lc.TurnOff();
            this.PlaceObjectTo(new int2(0, 0), lc);

            StaticObjects = new MyStaticObject[] { dc, door, door2, dc2, l, lc };
        }
    }

    /// <summary>
    /// end of custom map definitions
    /// </summary>
    public class Tale
    {
        public MyGridWorld.MyGraphicsPrototype Graphics { get; set; }
        public Boolean IsObstacle { get; set; }
        public List<MyStaticObject> Objects { get; set; }
    }

    public abstract class AbstractTwoStateObject : MyStaticObject
    {
        public MyGridWorld.MyGraphicsPrototype OnGraphics { get; private set; }
        public MyGridWorld.MyGraphicsPrototype OffGraphics { get; private set; }

        public void UpdateGraphics(MyGridWorld.MyGraphicsPrototype off_g, MyGridWorld.MyGraphicsPrototype on_g)
        {
            this.OnGraphics = on_g;
            this.OffGraphics = off_g;
            if (this.IsOn())
            {
                this.Graphics = on_g;
            }
            else
            {
                this.Graphics = off_g;
            }
        }

        public abstract Boolean IsOn();
        public abstract void TurnOn();
        public abstract void TurnOff();

        public void changeState()
        {
            if (this.IsOn())
            {
                this.TurnOff();
            }
            else
            {
                this.TurnOn();
            }
        }
    }

    public class MyDoor : AbstractTwoStateObject
    {
        public static readonly float DOOR_C_W = 1.0f;
        public static readonly float DOOR_O_W = 0.0f;

        public MyDoor()
            : base()
        {
            this.properties.Add(WEIGHT_KEY, DOOR_C_W);
        }

        public override Boolean IsOn()
        {
            return this.GetWeight() == DOOR_C_W;
        }

        public override void TurnOn()
        {
            this.properties[WEIGHT_KEY] = DOOR_C_W;
            this.Graphics = OnGraphics;
        }

        public override void TurnOff()
        {
            this.properties[WEIGHT_KEY] = DOOR_O_W;
            this.Graphics = OffGraphics;
        }
    }

    public class Lights : AbstractTwoStateObject
    {
        public static readonly String LIGHTS_STATE_KEY = "L";//original: "light"
        public static readonly float LIGHTS_ON = 1.0f;
        public static readonly float LIGHTS_OFF = 0.0f;

        public Lights()
            : base()
        {
            this.SetWeight(0);  // agent can step on the light
            this.properties.Add(LIGHTS_STATE_KEY, LIGHTS_ON);
        }

        public override void TurnOff()
        {
            this.properties[LIGHTS_STATE_KEY] = LIGHTS_OFF;
            this.Graphics = this.OffGraphics;
        }

        public override void TurnOn()
        {
            this.properties[LIGHTS_STATE_KEY] = LIGHTS_ON;
            this.Graphics = this.OnGraphics;
        }

        public override Boolean IsOn()
        {
            return this.properties[LIGHTS_STATE_KEY] == LIGHTS_ON;
        }
    }

    // for parsing of graphics..
    public class DoorControl : TwoStateObjectControl { }
    public class LightsControl : TwoStateObjectControl { }




    // by default the action will flip the state of all owned objects
    //public class TwoStateObjectControl : StaticObject
    public class TwoStateObjectControl : AbstractTwoStateObject
    {
        public static readonly String SWITCH_STATE_KEY = "SW";
        public static readonly float SWITCH_ON = 1.0f;
        public static readonly float SWITCH_OFF = 0.0f;

        public List<AbstractTwoStateObject> myObjects;

        public TwoStateObjectControl()
        {
            this.SetWeight(0);  // agent can step on the switch
            myObjects = new List<AbstractTwoStateObject>();
            this.properties.Add(SWITCH_STATE_KEY, SWITCH_ON);
        }

        public void applyPressAction()
        {
            for (int i = 0; i < myObjects.Count; i++)
            {
                myObjects[i].changeState();
            }
            base.changeState();
        }

        public override Boolean IsOn()
        {
            return this.properties[SWITCH_STATE_KEY] == SWITCH_ON;
        }

        public override void TurnOn()
        {
            this.properties[SWITCH_STATE_KEY] = SWITCH_ON;
            this.Graphics = this.OnGraphics;
        }

        public override void TurnOff()
        {
            this.properties[SWITCH_STATE_KEY] = SWITCH_OFF;
            this.Graphics = this.OffGraphics;

        }
    }


    public class MyStaticObject
    {
        public static readonly String WEIGHT_KEY = "S";// original: "weight";
        public static readonly float DEF_W;
        public static readonly float AGENT_W = 0.5f;    // weights

        public MyGridWorld.MyGraphicsPrototype Graphics { get; set; }
        protected int2 m_position;
        private float m_weight = DEF_W;

        // my properties that can be "serializable" to the node's output
        public Dictionary<String, float> properties = new Dictionary<String, float>();

        // another object in the map that I can control
        protected MyStaticObject myTarget;

        public int2 GetPosition()
        {
            return this.m_position;
        }

        public void setPosition(int2 pos)
        {
            this.m_position = pos;
        }

        public void SetWeight(float weight)
        {
            this.m_weight = weight;
        }

        public float GetWeight()
        {
            if (properties.ContainsKey(WEIGHT_KEY))
            {
                return properties[WEIGHT_KEY];
            }
            return m_weight;
        }
    }

    public class MyObjectProperty
    {
        public String key;  // name of the property
        public float value;

        public MyObjectProperty(String key, float value)
        {
            this.key = key;
            this.value = value;
        }
    }

    public class MyMovingObject : MyStaticObject
    {
        public int CurrentAction { get; private set; }

        public void MoveUp()
        {
            m_position.y++;
        }

        public void MoveDown()
        {
            m_position.y--;
        }

        public void MoveLeft()
        {
            m_position.x--;
        }
        public void MoveRight()
        {
            m_position.x++;
        }
    }

    public interface IWorldParser
    {
        void registerGraphics(
            MyGridWorld.MyGraphicsPrototype empty_g,
            MyGridWorld.MyGraphicsPrototype obstacle_g,
            MyGridWorld.MyGraphicsPrototype agent_g,
            MyGridWorld.MyGraphicsPrototype doorOpened_g,
            MyGridWorld.MyGraphicsPrototype doorClosed_g,
            MyGridWorld.MyGraphicsPrototype doorControl_g,
            MyGridWorld.MyGraphicsPrototype doorControlOff_g,
            MyGridWorld.MyGraphicsPrototype lightsControl_g,
            MyGridWorld.MyGraphicsPrototype lightsControlOff_g,
            MyGridWorld.MyGraphicsPrototype lightsOff_g,
            MyGridWorld.MyGraphicsPrototype lightsOn_g);

        Tale[,] GetTales();

        MyMovingObject[] GetMovingObjects();

        MyStaticObject[] GetStaticObjects();

        MyMovingObject GetAgent();

        void PlaceObjectTo(int2 pos, MyStaticObject obj);

        int GetWidth();
        int GetHeight();
        int[] GetArray();

    }

    public abstract class AbstractMap : IWorldParser
    {
        public static readonly int A = 2;               // agent in the map

        protected int[] m_array;
        protected int m_width = 0, m_height = 0;

        public Tale[,] Tiles { get; private set; }
        public MyMovingObject Agent { get; private set; }
        public MyStaticObject[] StaticObjects { get; set; }
        public MyMovingObject[] MovingObjects { get; set; }

        public AbstractMap()
        {
            this.DefineArray();
            this.ParseTiles();
            this.DefineMovingObjects();
            this.DefineStaticObjects();
        }

        public int[] GetArray()
        {
            return m_array;
        }

        public int GetWidth()
        {
            return m_width;
        }

        public int GetHeight()
        {
            return m_height;
        }

        public void registerGraphics(
            MyGridWorld.MyGraphicsPrototype empty_g,
            MyGridWorld.MyGraphicsPrototype obstacle_g,
            MyGridWorld.MyGraphicsPrototype agent_g,
            MyGridWorld.MyGraphicsPrototype doorOpened_g,
            MyGridWorld.MyGraphicsPrototype doorClosed_g,
            MyGridWorld.MyGraphicsPrototype doorControl_g,
            MyGridWorld.MyGraphicsPrototype doorControlOff_g,
            MyGridWorld.MyGraphicsPrototype lightsControl_g,
            MyGridWorld.MyGraphicsPrototype lightsControlOff_g,
            MyGridWorld.MyGraphicsPrototype lightsOff_g,
            MyGridWorld.MyGraphicsPrototype lightsOn_g)
        {
            Agent.Graphics = agent_g;

            // register all tales' graphics
            for (int j = 0; j < m_height; j++)
            {
                for (int i = 0; i < m_width; i++)
                {
                    if (Tiles[i, j].IsObstacle)
                    {
                        Tiles[i, j].Graphics = empty_g;
                    }
                    else
                    {
                        Tiles[i, j].Graphics = obstacle_g;
                    }
                }
            }

            // all static objects
            MyStaticObject tmp;
            for (int i = 0; i < StaticObjects.Length; i++)
            {
                tmp = StaticObjects[i];
                if (tmp is DoorControl)
                {
                    //tmp.graphics = doorControl_g;
                    ((AbstractTwoStateObject)tmp).UpdateGraphics(doorControl_g, doorControlOff_g);
                }
                else if (tmp is LightsControl)
                {
                    //tmp.graphics = lightsControl_g;
                    ((AbstractTwoStateObject)tmp).UpdateGraphics(lightsControl_g, lightsControlOff_g);
                }
                else if (tmp is MyDoor)
                {
                    ((AbstractTwoStateObject)tmp).UpdateGraphics(doorOpened_g, doorClosed_g);
                }
                else if (tmp is Lights)
                {
                    ((AbstractTwoStateObject)tmp).UpdateGraphics(lightsOn_g, lightsOff_g);
                }
            }

        }

        // define the 1D vector storing obstacles and agent, also set width and height
        protected abstract void DefineArray();
        protected abstract void DefineStaticObjects();
        protected abstract void DefineMovingObjects();

        private void ParseTiles()
        {
            Tiles = new Tale[m_width, m_height];
            for (int j = 0; j < m_height; j++)
            {
                for (int i = 0; i < m_width; i++)
                {
                    if (m_array[i + j * m_width] == 1)
                    {
                        Tiles[i, j] = new Tale()
                        {
                            IsObstacle = true,
                        };
                    }
                    else
                    {
                        Tiles[i, j] = new Tale()
                        {
                            IsObstacle = false,
                            Objects = new List<MyStaticObject>(),
                        };
                    }
                    if (m_array[i + j * m_width] == A)
                    {
                        this.PlaceAgentTo(i, j);
                    }
                }
            }

            if (Agent == null)
            {
                Console.WriteLine("Warning: no agent found in the map, placing it to [0,0]");
                this.PlaceAgentTo(0, 0);
            }
        }

        public void PlaceObjectTo(int2 pos, MyStaticObject obj)
        {
            obj.setPosition(pos);
            this.DeleteObstacle(pos.x, pos.y);
            Tiles[pos.x, pos.y].Objects.Add(obj);
        }

        private void PlaceAgentTo(int x, int y)
        {
            Agent = new MyMovingObject();
            Agent.SetWeight(MyStaticObject.AGENT_W);
            Agent.setPosition(new int2(x, y));
            this.DeleteObstacle(x, y);
        }

        private void DeleteObstacle(int x, int y)
        {
            // change obstacle to empty, if needed
            if (Tiles[x, y].IsObstacle)
            {
                Tiles[x, y].IsObstacle = false;
                Tiles[x, y].Objects = new List<MyStaticObject>();
            }
        }

        public MyMovingObject GetAgent()
        {
            return Agent;
        }

        public Tale[,] GetTales()
        {
            return Tiles;
        }

        public MyMovingObject[] GetMovingObjects()
        {
            return MovingObjects;
        }

        public MyStaticObject[] GetStaticObjects()
        {
            return StaticObjects;
        }
    }
}
