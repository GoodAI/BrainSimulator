#version 150

in vec4 in_position;
in vec3 in_color;

out vec4 out_color;

void main()
{
	gl_Position = in_position;
	out_color = vec4(in_color, 1);
}
