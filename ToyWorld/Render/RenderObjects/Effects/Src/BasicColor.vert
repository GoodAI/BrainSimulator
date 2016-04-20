#version 330

layout(location=0) in vec2 v_position;
layout(location=7) in vec3 v_color;

out vec4 f_color;

void main()
{
	gl_Position = vec4(v_position, 0, 1);
	f_color = vec4(v_color, 1);
}
