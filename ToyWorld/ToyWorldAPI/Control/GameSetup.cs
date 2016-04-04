using System.IO;
using System.Runtime.InteropServices;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// Setup object for ToyWorld
    /// </summary>
    public sealed class GameSetup
    {
        /// <summary>
        /// The path to a .TMX (fresh world) or a world save.
        /// </summary>
        public StreamReader SaveFile { get; private set; }

        /// <summary>
        /// Csv table containing columns "NameOfTile", "PositionInTileset" and "IsDefault"
        /// One TileName can have more PositionInTileset numbers, so object can behave based on common
        /// implementation but looks different according to position.
        /// 
        /// <p>(string) Name of tile must match name of implemented object.</p>
        /// <p>(int) Position in tileset is index of tile in png file with pictures of tiles and objects.
        /// It should match index in Tiled</p>
        /// <p>(0|1) . If is some tile default, when new tile is created, this TileNumber will be used.</p>
        /// </summary>
        public StreamReader TilesetFile { get; private set; }

        /// <summary>
        /// Setup object for ToyWorld constructor.
        /// </summary>
        /// <param name="saveFile">Tmx file created in Tiled editor or save file</param>
        /// <param name="tilesetFile">Csv table containing columns "NameOfTile", "PositionInTileset" and "IsDefault"</param>
        public GameSetup(StreamReader saveFile, StreamReader tilesetFile)
        {
            SaveFile = saveFile;
            TilesetFile = tilesetFile;
        }

        // TODO: more setup
    }
}
