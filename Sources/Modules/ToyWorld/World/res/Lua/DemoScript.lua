-- to execute, type "execute C:/folder/subdolfer/script.lua"
-- execute D:/BrainSimInternal/Sources/Modules/ToyWorld/World/res/Lua/DemoScript.lua

local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

function whichRoom(x,y)
	if x < 6.5 then
		if y < 6.5 then
			return 1
		else
			return 4
		end
	else
		if y < 6.5 then
			return 2
		else
			return 3
		end
	end
end

function backDoor(room)
	if(room == 1) then return 3,6
	elseif(room == 2) then return 6,3
	elseif(room == 3) then return 9,6
	elseif(room == 4) then return 6,9
	else return nil end
end


for i=1,10000 do
	local v = am:WhereIsObject("Ball")
	local room = whichRoom(v.X,v.Y)
	local x,y = backDoor(room)
	am:CreateTile("Wall","Obstacle",x,y)
	sleep(10)
	
	am:DestroyTile("Obstacle",x,y)
	sleep(2)
end