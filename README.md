# PixelArtPipeline for Roblox BedWars

## Overview
This project converts images into Lua tables for use with the BedWars Scripting Toolkit extension in VS Code. It enables you to create pixel art or custom block layouts for Roblox BedWars using your own images.

## Features
- Convert PNG/JPG images to Lua tables for BedWars
- Optional edge outlining and smoothing (configurable)
- Fast, vectorized color matching
- Output is always 512x512 for BedWars compatibility

## Requirements
- **Python 3.8+**
- **VS Code** with the [BedWars Scripting Toolkit extension](https://marketplace.visualstudio.com/items?itemName=easy-games.bedwars-scripting-toolkit) (required for syncing Lua tables to Roblox)
- See `requirements.txt` for Python dependencies

## Setup
1. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Install the BedWars Scripting Toolkit extension in VS Code.
3. Place your input image in the `src/` folder as `input.png` or `input.jpg`.

## Usage
Run the script from the project root:
```sh
python src/main.py
```

- Configure options (edge outlining, blur level) at the top of `main.py`.
- The output Lua table will be saved as `src/output.lua`.
- Sync this project to Bedwars (Create a custom, preferably void (squads) and open the scripts tab)
- Run `build_image.lua` in bedwars, and wait 3 seconds.

## Notes
- Only PNG and JPG/JPEG images are supported.
- Output is always 512x512 for BedWars compatibility.
- The `.gitignore` is set to ignore input/output files and VS Code settings.

## License
MIT License
