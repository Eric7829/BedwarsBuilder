-- Load the output data from the Python conversion
local imageData = require("output")

-- Block ID to ItemType mapping
-- Note: Some blocks may not exist in ItemType - adjust as needed
local BLOCK_TYPES = {
    [1] = ItemType.STONE_BRICK,
    [2] = ItemType.OBSIDIAN,
    [3] = ItemType.WOOL_WHITE,
    [4] = ItemType.ANDESITE,
    [5] = ItemType.CLAY_RED,
    [6] = ItemType.FISHERMAN_CORAL, -- CORAL_BLOCK IN ORIGINAL CODE
    [7] = ItemType.WOOD_PLANK_OAK,
    [8] = ItemType.WOOD_PLANK_BIRCH,
    [9] = ItemType.WOOD_PLANK_SPRUCE,
    [10] = ItemType.DIAMOND_BLOCK,
    [11] = ItemType.SAND,
    [12] = ItemType.PURPLE_LUCKY_BLOCK, 
    [13] = ItemType.WOOL_RED,
    [14] = ItemType.WOOL_GREEN,
    [15] = ItemType.WOOL_YELLOW,
    [16] = ItemType.WOOL_BLUE,
    [17] = ItemType.WOOL_CYAN,
    [18] = ItemType.WOOL_PINK,
    [19] = ItemType.WOOL_ORANGE,
    [20] = ItemType.WOOL_PURPLE,
    [21] = ItemType.BLASTPROOF_CERAMIC, 
    [22] = ItemType.CLAY_BLACK,
    [23] = ItemType.CLAY_LIGHT_GREEN,
    [24] = ItemType.CLAY_TAN,
    [25] = ItemType.CLAY_WHITE,
    [26] = ItemType.LUCKY_BLOCK,
    [27] = ItemType.DIORITE,
    [28] = ItemType.CLAY_DARK_BROWN,
    [29] = ItemType.CLAY_BLUE,
    [30] = ItemType.ICE,
    [31] = ItemType.CLAY_DARK_GREEN,
    [32] = ItemType.CONCRETE_GREEN,
    [33] = ItemType.CLAY_PURPLE,
    [34] = ItemType.MARBLE_PILLAR,
    [35] = ItemType.CLAY,
    [36] = ItemType.MARBLE,
    [37] = ItemType.IRON_BLOCK,
    [38] = ItemType.SANDSTONE_SMOOTH,
    [39] = ItemType.RED_SAND
}

-- Debugging: Check for missing ItemType mappings
for blockId, blockType in pairs(BLOCK_TYPES) do
    if not blockType then
        print("Debug: Missing ItemType for block ID:", blockId)
    end
end

-- Get all players
local players = PlayerService.getPlayers()
if #players == 0 then
    print("No players found!")
    return
end

MessageService.broadcast("Starting pixel art construction...")
task.wait(3)  -- Wait a few moments to ensure player entities are loaded

-- Get the first player's entity
local player = players[1]
local entity = player:getEntity()
if not entity then
    print("Player has no entity!")
    return
end

-- Get the player's position as the top-left corner
local startPosition = entity:getPosition()
print("Starting position:", startPosition)

-- Destroy the block under the player
local blockUnder = startPosition - Vector3.new(0, 5, 0)
BlockService.destroyBlock(blockUnder)
print("Destroyed block under player")

-- Build the pixel art
-- Each block is 1 units apart
-- X direction = horizontal (width)
-- Z direction = depth (height of image)
-- Y stays constant (no verticality)

local BLOCK_SPACING = 2.5
local height = #imageData  -- Number of rows
local width = #imageData[1]  -- Number of columns

print("Building " .. width .. "x" .. height .. " pixel art...")

local blocksPlaced = 0
local blocksFailed = 0

-- Loop through each pixel in the image
for z = 1, height do
    for x = 1, width do
        -- Get the block ID for this pixel
        local blockId = imageData[z][x]
        local blockType = BLOCK_TYPES[blockId]

        if blockType then
            -- Calculate position: start from player position, offset by X and Z
            -- Using (x-1) and (z-1) so the first pixel is at the player's position
            local blockPosition = Vector3.new(
                startPosition.X + (x - 1) * BLOCK_SPACING,
                startPosition.Y,  -- Same Y level (no verticality)
                startPosition.Z + (z - 1) * BLOCK_SPACING
            )

            -- Place the block
            local success = BlockService.placeBlock(blockType, blockPosition)
            if success then
                blocksPlaced = blocksPlaced + 1
            else
                blocksFailed = blocksFailed + 1
            end
        else
            print("Warning: Unknown block ID:", blockId)
            blocksFailed = blocksFailed + 1
        end
    end

    -- Progress update every 50 rows
    if z % 50 == 0 then
        print("Progress: " .. z .. "/" .. height .. " rows completed")
    end

    -- Small delay to prevent overwhelming the server
    task.wait(0.01)
end

print("Pixel art construction complete!")
print("Blocks placed: " .. blocksPlaced)
print("Blocks failed: " .. blocksFailed)
print("Total dimensions: " .. width .. " blocks wide x " .. height .. " blocks deep")
print("Physical size: " .. (width * BLOCK_SPACING) .. " units x " .. (height * BLOCK_SPACING) .. " units")
