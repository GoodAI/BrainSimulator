#version 330


uniform mat4 mw   = mat4(1);
uniform mat4 mvp  = mat4(1);


layout(location=0) in vec2 v_position;

out vec2 f_worldPos;


void main()
{
	vec4 pos4 = vec4(v_position, 0, 1);

	f_worldPos = (mw * pos4).xy;
	gl_Position = mvp * pos4;
}
