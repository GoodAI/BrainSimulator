using System.IO;

namespace ToyWorldTests
{

    internal static class FileStreams
    {
        public static Stream GetTmxMemoryStream()
        {
            var ms = new MemoryStream();
            const string fileString =
                @"<?xml version=""1.0"" encoding=""UTF-8""?>
                <map version=""1.0"" orientation=""orthogonal"" renderorder=""right-down"" width=""3"" height=""3"" tilewidth=""16"" tileheight=""16"" nextobjectid=""19"">
                        <tileset firstgid=""1"" source=""roguelikeSheet_summer.tsx""/>
                        <layer name=""Background"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    16,17,16,
                    16,16,16,
                    16,16,16
                    </data>
                        </layer>
                        <layer name=""OnBackground"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    0,0,0,
                    91,0,0,
                    0,0,0
                    </data>
                        </layer>
                        <layer name=""Path"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    0,0,0,
                    0,136,0,
                    0,0,0
                    </data>
                        </layer>
                        <layer name=""OnGroundInteractable"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    0,0,196,
                    0,0,0,
                    0,0,0
                    </data>
                        </layer>
                        <layer name=""ObstacleInteractable"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    0,0,0,
                    0,0,0,
                    0,0,316
                    </data>
                        </layer>
                        <layer name=""Obstacle"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    0,0,0,
                    0,0,256,
                    271,256,0
                    </data>
                        </layer>
                        <objectgroup name=""Object"">
                          <object id=""16"" name=""Pingu"" type=""Avatar"" gid=""496"" x=""9"" y=""22"" width=""16"" height=""16"" rotation=""160"">
                           <properties>
                            <property name=""ForwardSpeed"" value=""0.5""/>
                           </properties>
                          </object>
                        </objectgroup>
                        <layer name=""Foreground"" width=""3"" height=""3"">
                        <data encoding=""csv"">
                    0,0,0,
                    0,0,0,
                    0,0,0
                    </data>
                        </layer>
                        <objectgroup name=""ForegroundObject""/>
                </map>
                ";
            WriteToMemoryStream(ms, fileString);
            ms.Position = 0;
            return ms;
        }

        public static Stream GetTilesetTableMemoryStream()
        {
            var ms = new MemoryStream();
            const string fileString = 
                @"Layer;NameOfTile;PositionInTileset;IsDefault;Note
                Background;Background;16;1;Grass
                ;Background;17;0;Tile
                ;Background;18;0;Small rocks
                ;Background;19;0;Larger rocks
                ;Background;31;0;Grass
                ;Background;32;0;Tile
                ;Background;33;0;Small rocks
                ;Background;34;0;Larger rocks
                OnBackground;OnBackground;76;1;Small plant
                ;OnBackground;91;0;Small plant
                Paths;Path;136;1;
                ;Path;137;0;
                OnGroundInteractable;Water;196;1;
                ;FireplaceBurning;197;1;
                ;Fireplace;198;1;
                ;Water;211;0;
                Obstacle;Obstacle;256;1;Non destroyable Wall
                ;AppleTree;257;1;
                ;PearTree;258;1;
                ;ConeTree;259;1;
                ;Wall;271;1;
                ObstacleInteractable;Apple;316;1;
                ;Pear;317;1;
                ;Cone;318;1;
                ;WoodenDoor;319;1;
                ;IronDoor;320;1;
                ;WoodenDoorLocked;321;1;
                ;IronBars;322;1;
                ;LeverPosition1;323;1;
                ;LeverPosition2;324;1;
                ;LeverPosition3;325;1;
                ;Pickaxe;331;1;
                ;OpenedWoodeDoor;334;1;
                ;OpenedIronDoor;335;1;
                ";

            WriteToMemoryStream(ms, fileString);
            ms.Position = 0;
            return ms;
        }

        public static Stream FullTmxFileStream()
        {
            return new FileStream(@"\TestFiles\mockup999_pantry_world.tmx", FileMode.Open);
        }

        private static void WriteToMemoryStream(MemoryStream memoryStream, string stringToWrite)
        {
            var stringBytes = System.Text.Encoding.UTF8.GetBytes(stringToWrite);
            memoryStream.Write(stringBytes, 0, stringBytes.Length);
        }
    }

}