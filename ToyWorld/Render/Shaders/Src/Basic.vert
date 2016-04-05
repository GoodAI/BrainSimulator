#version 330

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_color;

out vec4 color;

void main()
{
	gl_Position = vec4(in_position, 1);
	color = vec4(in_color, 1);
}
