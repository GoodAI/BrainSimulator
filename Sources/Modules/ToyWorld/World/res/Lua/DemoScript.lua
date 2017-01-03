-- to execute, type "execute C:/folder/subdolfer/script.lua"
-- execute D:/BrainSimInternal/Sources/Modules/ToyWorld/World/res/Lua/DemoScript.lua

local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

for i=1,3 do
	am:CreateTile("Wall","Obstacle",6,3)
	am:CreateTile("Wall","Obstacle",3,6)
	am:CreateTile("Wall","Obstacle",6,9)
	am:CreateTile("Wall","Obstacle",9,6)
	sleep(2)
	am:DestroyTile("Obstacle",6,3)
	am:DestroyTile("Obstacle",3,6)
	am:DestroyTile("Obstacle",6,9)
	am:DestroyTile("Obstacle",9,6)
	sleep(4)
end