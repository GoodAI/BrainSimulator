-- to execute, type "execute C:/folder/subdolfer/script.lua"
-- execute D:/BrainSimInternal/Sources/Modules/ToyWorld/World/res/Lua/DemoScript.lua

local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

function backDoor(room)
	if(room == "1") then return 3,6
	elseif(room == "2") then return 6,3
	elseif(room == "3") then return 9,6
	elseif(room == "4") then return 6,9
	else return nil end
end

local ball = am:GetObject("Ball")
for i=1,10 do
	lc:Print("Cyclus nr.: " .. i)
	local v = ball.Position
	local room = am:InRoom(ball)
	if(room == nil) then
		lc:Print("Ball in no room!")
	else
		local x,y = backDoor(room)
		lc:Print("Current room: " .. tostring(room))
		am:CreateTile("Wall","Obstacle",x,y)
		sleep(4)
		
		am:DestroyTile("Obstacle",x,y)
		sleep(2)
	end
end