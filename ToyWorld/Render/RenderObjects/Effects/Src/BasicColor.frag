#version 330

in vec4 f_color;

layout(location=0) out vec4 out_color;

void main()
{
	out_color = f_color;
}
