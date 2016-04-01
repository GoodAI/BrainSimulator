namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class GameSetup
    {
        /// <summary>
        /// The path to a .TMX (fresh world) or a world save.
        /// </summary>
        public string PathToSaveFile { get; set; }

        public string pathToTilesetFile  { get; set; }

        public GameSetup(string pathToSaveFile, string pathToTilesetFile = @"World\GameActors\Tiles\Tilesets\TilesetTable.csv")
        {
            PathToSaveFile = pathToSaveFile;
            this.pathToTilesetFile = pathToTilesetFile;
        }

        // TODO: more setup
    }
}
